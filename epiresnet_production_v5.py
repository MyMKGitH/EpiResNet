from __future__ import annotations

"""
EpiResNet v5 — production-ready multimodal AMR modelling pipeline.

Core model retained from the reference design:
    ESM-2 + LoRA protein encoder
    RDKit molecule graph + edge-aware GAT
    Bidirectional residue<->atom cross-attention
    Binary AMR classification + auxiliary MIC task

Major scientific upgrades:
    * MIC is represented as exact, left-censored, right-censored, or finite interval log2(MIC).
    * Binary S/R labels are derived only when the MIC interval is definitively
      on one side of organism/antibiotic-specific breakpoints.
    * Censored MICs are NOT treated as exact regression targets.
    * Missing breakpoints may remain for MIC regression but are masked from the
      binary objective.
    * Split groups are connected components over leakage entities, preventing
      indirect leakage through shared isolates or cold-start entities.
    * Exact duplicate assay observations are kept in the same partition.
    * Supports protein, antibiotic, scaffold, species, mechanism and joint
      cold-start evaluation.
    * Validation-only threshold selection with frozen checkpoint threshold and comprehensive metrics.
    * Best-checkpoint selection uses validation AUPRC, never test data.
    * Parameter-group learning rates for LoRA, graph encoder and heads.
    * AMP uses the current torch.amp API.
    * Artifact manifest, split files, metrics and configuration are exported.

Input columns expected by default:
    protein_sequence
    smiles
    mic_value
    mic_unit
    mic_operator
    [mic_lower_value, mic_upper_value]  # required only when mic_operator=interval/between
    susceptible_breakpoint
    resistant_breakpoint
    breakpoint_source
    antibiotic_id
    antibiotic_name
    drug_scaffold
    organism_species
    resistance_mechanism
    protein_cluster
    isolate_id

The breakpoint columns must come from a curator-specified standard/version. The
code never invents clinical breakpoints.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

RDLogger.DisableLog("rdApp.warning")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    # Model
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    seed: int = 42
    deterministic: bool = True
    max_seq_len: Optional[int] = None  # None => tokenizer/model maximum
    truncate_long_sequences: bool = False
    sequence_overflow_strategy: str = "error"  # error | n_terminal | center

    # Molecular graph
    atom_hidden: int = 96
    atom_heads: int = 4
    graph_dropout: float = 0.10

    # Fusion
    fusion_heads: int = 4
    fusion_dropout: float = 0.10
    fusion_ffn_dim: int = 512
    classifier_dropout: float = 0.25

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.10

    # Multi-task objective
    mic_loss_weight: float = 0.25
    mic_sigma_floor: float = 0.05

    # Optimization
    epochs: int = 30
    batch_size: int = 8
    grad_accum_steps: int = 1
    lr_lora: float = 1e-4
    lr_graph: float = 3e-4
    lr_head: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    patience: int = 6
    min_delta: float = 1e-4

    # Data / target
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    allow_missing_breakpoints: bool = True
    exclude_nonpositive_mic: bool = True
    derive_bemis_murcko_scaffold: bool = True

    mic_value_column: str = "mic_value"
    mic_unit_column: str = "mic_unit"
    mic_operator_column: str = "mic_operator"
    mic_lower_value_column: str = "mic_lower_value"
    mic_upper_value_column: str = "mic_upper_value"
    susceptible_breakpoint_column: str = "susceptible_breakpoint"
    resistant_breakpoint_column: str = "resistant_breakpoint"
    breakpoint_source_column: str = "breakpoint_source"
    breakpoint_unit_column: str = "breakpoint_unit"
    antibiotic_id_column: str = "antibiotic_id"
    antibiotic_name_column: str = "antibiotic_name"
    drug_scaffold_column: str = "drug_scaffold"
    species_column: str = "organism_species"
    mechanism_column: str = "resistance_mechanism"
    protein_cluster_column: str = "protein_cluster"
    isolate_id_column: str = "isolate_id"
    assay_id_column: str = "assay_id"

    split_strategy: str = "protein_cold"
    split_retries: int = 300
    require_binary_classes_in_all_splits: bool = True
    split_prevalence_tolerance: float = 0.10
    split_size_tolerance: float = 0.10
    minimum_split_rows: int = 10

    # Runtime
    num_workers: int = 0
    mixed_precision: str = "bf16"  # off | bf16 | fp16 | auto
    output_dir: str = "artifacts/epiresnet_v5"

    # Classification threshold
    threshold_metric: str = "mcc"  # mcc | f1 | balanced_accuracy
    calibrate_probabilities: bool = True
    calibration_max_iter: int = 100

    # Supported split strategies
    allowed_split_strategies: Tuple[str, ...] = (
        "random_group",
        "protein_cold",
        "antibiotic_cold",
        "scaffold_cold",
        "species_cold",
        "mechanism_cold",
        "joint_cold",
    )


# =============================================================================
# REPRODUCIBILITY / UTILITIES
# =============================================================================

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def normalize_missing(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "na", "n/a"} else s


def normalize_entity(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_missing(value)).strip()


def normalize_multilabel(value: Any) -> str:
    s = normalize_entity(value)
    if not s:
        return ""
    # Semicolon is the preferred multi-mechanism delimiter; comma is accepted
    # only as a fallback for curated data.
    parts = [p.strip() for p in re.split(r";", s) if p.strip()]
    if len(parts) == 1 and "," in parts[0]:
        parts = [p.strip() for p in parts[0].split(",") if p.strip()]
    return ";".join(sorted(set(parts), key=str.lower))


# =============================================================================
# MIC REPRESENTATION / TARGET DEFINITION
# =============================================================================

MIC_UNIT_TO_MG_L: Dict[str, float] = {
    "mg/l": 1.0,
    "mg/liter": 1.0,
    "mg/litre": 1.0,
    "mg/ml": 1000.0,
    "g/l": 1000.0,
    "ug/ml": 1.0,
    "μg/ml": 1.0,
    "µg/ml": 1.0,
    "mcg/ml": 1.0,
    "ug/l": 0.001,
    "μg/l": 0.001,
    "µg/l": 0.001,
    "mcg/l": 0.001,
}

VALID_MIC_OPERATORS = {"=", "==", "<", "<=", ">", ">="}


def normalize_operator(op: Any) -> str:
    value = normalize_entity(op).replace("≤", "<=").replace("≥", ">=")
    if value == "==":
        return "="
    if value not in {"=", "<", "<=", ">", ">="}:
        raise ValueError(f"Unsupported MIC operator {op!r}. Expected one of {sorted(VALID_MIC_OPERATORS)}.")
    return value


def mic_to_mg_l(value: Any, unit: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MIC value {value!r} is not numeric") from exc
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"MIC value must be finite and > 0; got {value!r}")
    u = normalize_entity(unit).lower().replace(" ", "")
    if u not in MIC_UNIT_TO_MG_L:
        raise ValueError(f"Unsupported MIC unit {unit!r}")
    out = x * MIC_UNIT_TO_MG_L[u]
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"Normalized MIC is invalid: {out!r}")
    return float(out)


def breakpoint_to_mg_l(value: Any, unit: Any) -> float:
    """Normalize a clinical breakpoint to mg/L using the same unit map as MICs."""
    return mic_to_mg_l(value, unit)


def make_mic_log2_interval_from_values(lower_mg_l: float, upper_mg_l: float) -> Tuple[float, float, str]:
    """Convert a finite MIC interval [lower, upper] into log2 bounds."""
    lower_mg_l = float(lower_mg_l)
    upper_mg_l = float(upper_mg_l)
    if not (np.isfinite(lower_mg_l) and np.isfinite(upper_mg_l) and lower_mg_l > 0 and upper_mg_l > 0):
        raise ValueError("Finite MIC interval bounds must be positive and finite")
    if lower_mg_l > upper_mg_l:
        raise ValueError(f"MIC interval lower bound exceeds upper bound: {lower_mg_l} > {upper_mg_l}")
    return float(np.log2(lower_mg_l)), float(np.log2(upper_mg_l)), "exact" if lower_mg_l == upper_mg_l else "interval"


def make_mic_log2_interval(mic_mg_l: float, operator: str) -> Tuple[float, float, str]:
    """Return log2(MIC) lower/upper bounds and censor type.

    Exact: [x, x]
    Left-censored (<x, <=x): [-inf, x]
    Right-censored (>x, >=x): [x, +inf]

    The finite endpoint is treated as the observation limit, not as the exact
    latent MIC for censored observations.
    """
    x = float(np.log2(mic_mg_l))
    if operator == "=":
        return x, x, "exact"
    if operator in {"<", "<="}:
        return -math.inf, x, "left"
    if operator in {">", ">="}:
        return x, math.inf, "right"
    raise ValueError(f"Unexpected normalized MIC operator: {operator}")


def classify_mic_interval(
    lower_log2: float,
    upper_log2: float,
    censor_type: str,
    susceptible_bp_mg_l: float,
    resistant_bp_mg_l: float,
) -> int:
    """Definitive S/R classification from a MIC interval.

    Returns:
        0 = definitely susceptible
        1 = definitely resistant
       -1 = indeterminate/insufficiently bounded

    This intentionally does NOT label a censored observation as resistant just
    because its upper bound crosses R, or susceptible just because its lower
    bound crosses S. The entire feasible interval must lie on one side.
    """
    s = math.log2(float(susceptible_bp_mg_l))
    r = math.log2(float(resistant_bp_mg_l))
    if not (np.isfinite(s) and np.isfinite(r) and s < r):
        raise ValueError("Breakpoints must be finite, positive, and satisfy susceptible < resistant.")

    if censor_type in {"exact", "interval"}:
        if upper_log2 <= s:
            return 0
        if lower_log2 >= r:
            return 1
        return -1
    if censor_type == "left":
        # Feasible values are (-inf, upper]. Only an upper bound <= S is
        # sufficient for definitive S classification.
        return 0 if upper_log2 <= s else -1
    if censor_type == "right":
        # Feasible values are [lower, +inf]. Only a lower bound >= R is
        # sufficient for definitive R classification.
        return 1 if lower_log2 >= r else -1
    raise ValueError(f"Unsupported censor type {censor_type!r}")


def validate_breakpoint_metadata(row: pd.Series, cfg: Config) -> Tuple[Optional[float], Optional[float], str, str]:
    source = normalize_entity(row.get(cfg.breakpoint_source_column, ""))
    unit = normalize_entity(row.get(cfg.breakpoint_unit_column, "mg/L")) or "mg/L"
    s_raw = row.get(cfg.susceptible_breakpoint_column, np.nan)
    r_raw = row.get(cfg.resistant_breakpoint_column, np.nan)
    if pd.isna(s_raw) or pd.isna(r_raw):
        if cfg.allow_missing_breakpoints:
            return None, None, source, unit
        raise ValueError("Missing susceptible/resistant breakpoint while allow_missing_breakpoints=False")
    s = breakpoint_to_mg_l(s_raw, unit)
    r = breakpoint_to_mg_l(r_raw, unit)
    if not (np.isfinite(s) and np.isfinite(r) and s > 0 and r > 0 and s < r):
        raise ValueError(f"Invalid breakpoints after unit normalization: S={s_raw!r}, R={r_raw!r}, unit={unit!r}")
    if not source:
        warnings.warn("Breakpoint values supplied without breakpoint_source; provenance should be recorded.")
    return s, r, source, unit


# =============================================================================
# SEQUENCE / MOLECULE FEATURIZATION
# =============================================================================

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
ATOMIC_NUMBERS = tuple(range(1, 21)) + (26, 35, 53)
HYBRIDIZATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
    Chem.HybridizationType.UNSPECIFIED,
)
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
BOND_STEREO = tuple(
    getattr(Chem.rdchem.BondStereo, name)
    for name in ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS")
    if hasattr(Chem.rdchem.BondStereo, name)
)


def one_hot(value: Any, choices: Sequence[Any]) -> List[float]:
    return [float(value == c) for c in choices] + [float(value not in choices)]


def normalize_protein_sequence(sequence: Any) -> str:
    s = normalize_entity(sequence).replace(" ", "").upper()
    if not s:
        raise ValueError("Protein sequence is empty")
    invalid = sorted(set(s) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Unsupported amino-acid symbols: {invalid}")
    return s


def canonicalize_smiles(smiles: Any) -> Tuple[str, Chem.Mol]:
    s = normalize_entity(smiles)
    if not s:
        raise ValueError("SMILES is empty")
    mol = Chem.MolFromSmiles(s, sanitize=True)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol


def atom_features(atom: Chem.Atom) -> List[float]:
    atomic_num = atom.GetAtomicNum()
    degree = min(atom.GetDegree(), 5)
    formal_charge = atom.GetFormalCharge()
    charge_bucket = formal_charge if formal_charge in (-2, -1, 0, 1, 2) else 99
    hybrid = atom.GetHybridization()
    total_h = min(atom.GetTotalNumHs(), 4)
    chiral = int(atom.GetChiralTag())
    return (
        one_hot(atomic_num, ATOMIC_NUMBERS)
        + one_hot(degree, tuple(range(6)))
        + one_hot(charge_bucket, (-2, -1, 0, 1, 2))
        + one_hot(hybrid, HYBRIDIZATIONS)
        + one_hot(total_h, tuple(range(5)))
        + one_hot(chiral, tuple(range(4)))
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass() / 100.0),
            float(atom.GetNumRadicalElectrons()),
        ]
    )


def bond_features(bond: Chem.Bond) -> List[float]:
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + one_hot(bond.GetStereo(), BOND_STEREO)
    )


def feature_dimensions() -> Tuple[int, int]:
    sample_atom = Chem.MolFromSmiles("CC").GetAtomWithIdx(0)
    sample_bond = Chem.MolFromSmiles("CC").GetBondWithIdx(0)
    return len(atom_features(sample_atom)), len(bond_features(sample_bond))


def bemis_murcko_scaffold(smiles: str) -> str:
    """Canonical Bemis-Murcko scaffold for chemical cold-start splitting."""
    canonical, mol = canonicalize_smiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def smiles_to_graph(smiles: str) -> Data:
    canonical, mol = canonicalize_smiles(smiles)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Molecule produced no atoms: {smiles!r}")

    edges: List[List[int]] = []
    edge_attr: List[List[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edges.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])

    e_dim = len(bond_features(mol.GetBondWithIdx(0))) if mol.GetNumBonds() else feature_dimensions()[1]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        eattr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        eattr = torch.empty((0, e_dim), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=eattr, smiles=canonical)


# =============================================================================
# DATA VALIDATION / PREPARATION
# =============================================================================

REQUIRED_COLUMNS = {
    "protein_sequence",
    "smiles",
    "mic_value",
    "mic_unit",
    "mic_operator",
    "antibiotic_id",
    "antibiotic_name",
    "drug_scaffold",
    "organism_species",
    "resistance_mechanism",
    "protein_cluster",
    "isolate_id",
}


def check_required_columns(df: pd.DataFrame, cfg: Config) -> None:
    required = {
        "protein_sequence",
        "smiles",
        cfg.mic_value_column,
        cfg.mic_unit_column,
        cfg.mic_operator_column,
        cfg.antibiotic_id_column,
        cfg.antibiotic_name_column,
        cfg.species_column,
        cfg.mechanism_column,
        cfg.protein_cluster_column,
        cfg.isolate_id_column,
    }
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def validate_identifier_consistency(df: pd.DataFrame, cfg: Config) -> None:
    # Exact antibiotic IDs should not silently map to multiple curated names.
    pairs = df[[cfg.antibiotic_id_column, cfg.antibiotic_name_column]].copy()
    pairs = pairs.dropna().astype(str)
    conflicts = pairs.groupby(cfg.antibiotic_id_column)[cfg.antibiotic_name_column].nunique(dropna=True)
    bad = conflicts[conflicts > 1]
    if not bad.empty:
        raise ValueError(
            f"antibiotic_id maps to multiple antibiotic_name values: {bad.index.tolist()[:10]}"
        )

    if "smiles" in df.columns:
        mol_pairs = df[[cfg.antibiotic_id_column, "smiles"]].copy()
        mol_pairs = mol_pairs.dropna().astype(str)
        mol_pairs["canonical_smiles"] = mol_pairs["smiles"].map(lambda x: canonicalize_smiles(x)[0])
        structure_conflicts = mol_pairs.groupby(cfg.antibiotic_id_column)["canonical_smiles"].nunique()
        bad_structure = structure_conflicts[structure_conflicts > 1]
        if not bad_structure.empty:
            raise ValueError(
                f"antibiotic_id maps to multiple molecular structures: {bad_structure.index.tolist()[:10]}"
            )

    # Species/mechanism/protein-cluster identifiers are leakage-control metadata.
    # Empty values are rejected later for the selected split regime, but we flag
    # them here because silent missingness can create accidental random grouping.
    for col in [cfg.species_column, cfg.mechanism_column, cfg.protein_cluster_column, cfg.isolate_id_column]:
        if col in df.columns and df[col].map(normalize_entity).eq("").any():
            warnings.warn(f"Column {col!r} contains missing identifiers; strict cold-start splitting may fail.")


def parse_mic_observation(row: pd.Series, cfg: Config) -> Tuple[float, float, str, float]:
    """Parse exact/censored or explicit finite-interval MIC observations."""
    op = normalize_operator(row[cfg.mic_operator_column])
    if op in {"=", "<", "<=", ">", ">="}:
        mic = mic_to_mg_l(row[cfg.mic_value_column], row[cfg.mic_unit_column])
        lower, upper, censor = make_mic_log2_interval(mic, op)
        return lower, upper, censor, mic

    raise ValueError(f"Unsupported MIC operator: {op}")


def prepare_assay_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    check_required_columns(df, cfg)
    validate_identifier_consistency(df, cfg)
    out = df.copy().reset_index(drop=True)

    sequences: List[str] = []
    smiles: List[str] = []
    canonical_smiles: List[str] = []
    mic_mg_l: List[float] = []
    mic_lower: List[float] = []
    mic_upper: List[float] = []
    censor_types: List[str] = []
    labels: List[int] = []
    binary_valid: List[bool] = []
    sources: List[str] = []
    breakpoint_units: List[str] = []

    for i, row in out.iterrows():
        try:
            seq = normalize_protein_sequence(row["protein_sequence"])
            canonical, mol = canonicalize_smiles(row["smiles"])
            raw_op = normalize_entity(row[cfg.mic_operator_column]).replace("≤", "<=").replace("≥", ">=").lower()
            if raw_op in {"interval", "between", "[]", "range"}:
                if cfg.mic_lower_value_column not in out.columns or cfg.mic_upper_value_column not in out.columns:
                    raise ValueError(
                        "Interval MIC observation requires mic_lower_value and mic_upper_value columns."
                    )
                lower_mg_l = mic_to_mg_l(row[cfg.mic_lower_value_column], row[cfg.mic_unit_column])
                upper_mg_l = mic_to_mg_l(row[cfg.mic_upper_value_column], row[cfg.mic_unit_column])
                lower, upper, censor = make_mic_log2_interval_from_values(lower_mg_l, upper_mg_l)
                mic = math.sqrt(lower_mg_l * upper_mg_l)
            else:
                op = normalize_operator(row[cfg.mic_operator_column])
                mic = mic_to_mg_l(row[cfg.mic_value_column], row[cfg.mic_unit_column])
                lower, upper, censor = make_mic_log2_interval(mic, op)
            s_bp, r_bp, source, bp_unit = validate_breakpoint_metadata(row, cfg)
            if s_bp is not None and r_bp is not None:
                label = classify_mic_interval(lower, upper, censor, s_bp, r_bp)
            else:
                label = -1

            sequences.append(seq)
            smiles.append(normalize_entity(row["smiles"]))
            canonical_smiles.append(canonical)
            mic_mg_l.append(mic)
            mic_lower.append(lower)
            mic_upper.append(upper)
            censor_types.append(censor)
            labels.append(label)
            binary_valid.append(label in (0, 1))
            sources.append(source)
            breakpoint_units.append(bp_unit)
        except Exception as exc:
            raise ValueError(f"Row {i} failed validation: {exc}") from exc

    out["protein_sequence"] = sequences
    out["smiles"] = canonical_smiles
    out["mic_mg_l"] = mic_mg_l
    out["mic_log2_lower"] = mic_lower
    out["mic_log2_upper"] = mic_upper
    out["mic_censor_type"] = censor_types
    out["phenotype"] = np.asarray(labels, dtype=np.int8)
    out["binary_valid"] = np.asarray(binary_valid, dtype=bool)
    out["breakpoint_source_normalized"] = sources
    out["breakpoint_unit_normalized"] = breakpoint_units

    if cfg.drug_scaffold_column not in out.columns:
        if cfg.derive_bemis_murcko_scaffold:
            out[cfg.drug_scaffold_column] = ""
        else:
            raise ValueError(f"Missing required column: {cfg.drug_scaffold_column}")
    out[cfg.drug_scaffold_column] = out[cfg.drug_scaffold_column].map(normalize_entity)
    if cfg.derive_bemis_murcko_scaffold:
        derived_scaffolds = []
        for smi, supplied in zip(out["smiles"], out[cfg.drug_scaffold_column]):
            derived = bemis_murcko_scaffold(smi)
            if not derived:
                # Acyclic compounds have an empty Bemis-Murcko scaffold. Treat the
                # canonical molecule itself as the fallback chemical cold-start key,
                # rather than grouping all acyclic antibiotics into one artificial class.
                derived = "ACYCLIC::" + smi
            if supplied and supplied != derived:
                warnings.warn(
                    f"Supplied drug scaffold differs from derived Bemis-Murcko scaffold for {smi!r}; "
                    "using the derived scaffold for scaffold cold-start splitting."
                )
            derived_scaffolds.append(derived)
        out[cfg.drug_scaffold_column] = derived_scaffolds

    for col in [
        cfg.antibiotic_id_column,
        cfg.antibiotic_name_column,
        cfg.species_column,
        cfg.protein_cluster_column,
        cfg.isolate_id_column,
    ]:
        out[col] = out[col].map(normalize_entity)
    out[cfg.mechanism_column] = out[cfg.mechanism_column].map(normalize_multilabel)

    # The same assay observation should be inseparable across splits.
    hash_cols = [
        "protein_sequence",
        "smiles",
        cfg.antibiotic_id_column,
        cfg.species_column,
        cfg.mechanism_column,
        "mic_log2_lower",
        "mic_log2_upper",
        "mic_censor_type",
        "breakpoint_source_normalized",
        "breakpoint_unit_normalized",
    ]
    out["observation_hash"] = out[hash_cols].astype(str).agg("||".join, axis=1).map(sha256_text)

    # Remove exact duplicate assay records, but never collapse contradictory
    # measurements silently.
    before = len(out)
    replicate_counts = out["observation_hash"].value_counts().to_dict()
    conflict_fields = ["mic_log2_lower", "mic_log2_upper", "mic_censor_type", "phenotype"]
    conflicts = out.groupby("observation_hash")[conflict_fields].nunique(dropna=False)
    conflicting_hashes = conflicts[(conflicts > 1).any(axis=1)]
    if not conflicting_hashes.empty:
        raise ValueError(
            f"Found {len(conflicting_hashes)} duplicate observation groups with conflicting MIC/target fields. "
            "Resolve them in the source data instead of silently selecting one record."
        )
    out = out.drop_duplicates(subset=["observation_hash"], keep="first").reset_index(drop=True)
    out["replicate_count"] = out["observation_hash"].map(replicate_counts).astype(int)

    # Validate cold-start fields needed by the chosen strategy.
    if cfg.split_strategy not in cfg.allowed_split_strategies:
        raise ValueError(
            f"Unsupported split_strategy={cfg.split_strategy!r}; choose from {cfg.allowed_split_strategies}"
        )
    strategy_cols = split_entity_columns(cfg)
    missing_strategy_values = [c for c in strategy_cols if (out[c] == "").any()]
    if missing_strategy_values:
        raise ValueError(
            "Empty identifiers prevent a rigorous cold-start split: "
            + ", ".join(missing_strategy_values)
        )

    out = out.reset_index(drop=True)
    out.attrs["deduplicated_rows"] = int(before - len(out))
    return out


# =============================================================================
# LEAKAGE-SAFE CONNECTED-COMPONENT SPLITTING
# =============================================================================

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def split_entity_columns(cfg: Config) -> List[str]:
    mapping = {
        "random_group": [],
        "protein_cold": [cfg.protein_cluster_column],
        "antibiotic_cold": [cfg.antibiotic_id_column],
        "scaffold_cold": [cfg.drug_scaffold_column],
        "species_cold": [cfg.species_column],
        "mechanism_cold": [cfg.mechanism_column],
        "joint_cold": [
            cfg.protein_cluster_column,
            cfg.antibiotic_id_column,
            cfg.drug_scaffold_column,
            cfg.species_column,
            cfg.mechanism_column,
        ],
    }
    if cfg.split_strategy not in mapping:
        raise ValueError(f"Unknown split strategy {cfg.split_strategy!r}")
    return mapping[cfg.split_strategy]


def connected_component_groups(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Build true leakage components.

    Rows are connected when they share any mandatory integrity entity (isolate or
    exact observation) and, for the selected cold-start regime, when they share
    any cold-start entity. This is the transitive closure that simple string
    concatenation/grouping misses.
    """
    n = len(df)
    uf = UnionFind(n)
    keys_to_first: Dict[Tuple[str, str], int] = {}

    integrity_cols = [cfg.isolate_id_column, "observation_hash"]
    if cfg.assay_id_column in df.columns:
        integrity_cols.append(cfg.assay_id_column)
    cold_cols = split_entity_columns(cfg)
    all_cols = integrity_cols + cold_cols

    for col in all_cols:
        for i, value in enumerate(df[col].astype(str)):
            values = [value]
            if col == cfg.mechanism_column and value:
                values = [part.strip() for part in value.split(";") if part.strip()]
            for entity_value in values:
                key = (col, entity_value)
                if entity_value:
                    j = keys_to_first.get(key)
                    if j is None:
                        keys_to_first[key] = i
                    else:
                        uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    root_to_group: Dict[int, int] = {}
    groups = np.zeros(n, dtype=np.int64)
    next_group = 0
    for i, root in enumerate(roots):
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        groups[i] = root_to_group[root]
    return groups


def prevalence(df: pd.DataFrame) -> float:
    valid = df[df["binary_valid"]]
    return float(valid["phenotype"].mean()) if len(valid) else float("nan")


def split_overlap(a: pd.DataFrame, b: pd.DataFrame, columns: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for c in columns:
        if c == "resistance_mechanism":
            sa = {x.strip() for v in a[c].astype(str) for x in v.split(";") if x.strip()}
            sb = {x.strip() for v in b[c].astype(str) for x in v.split(";") if x.strip()}
        else:
            sa = set(a[c].astype(str))
            sb = set(b[c].astype(str))
        result[c] = len((sa & sb) - {""})
    return result


def split_is_valid(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> bool:
    splits = [train, val, test]
    if any(len(s) < cfg.minimum_split_rows for s in splits):
        return False
    if cfg.require_binary_classes_in_all_splits:
        if any(s["binary_valid"].sum() == 0 for s in splits):
            return False
        if any(s.loc[s["binary_valid"], "phenotype"].nunique() < 2 for s in splits):
            return False
    for entity_col in [cfg.isolate_id_column, "observation_hash", *split_entity_columns(cfg)]:
        if not entity_col:
            continue
        if split_overlap(train, val, [entity_col])[entity_col] > 0:
            return False
        if split_overlap(train, test, [entity_col])[entity_col] > 0:
            return False
        if split_overlap(val, test, [entity_col])[entity_col] > 0:
            return False
    return True


def split_score(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> float:
    target_prev = prevalence(pd.concat([train, val, test], ignore_index=True))
    vals = [prevalence(train), prevalence(val), prevalence(test)]
    if not np.isfinite(target_prev):
        return float("inf")
    prev_penalty = sum(abs(v - target_prev) for v in vals if np.isfinite(v))
    target_sizes = np.array([1 - cfg.val_fraction - cfg.test_fraction, cfg.val_fraction, cfg.test_fraction])
    actual_sizes = np.array([len(train), len(val), len(test)]) / max(1, len(train) + len(val) + len(test))
    size_penalty = float(np.abs(actual_sizes - target_sizes).sum())
    return prev_penalty / max(cfg.split_prevalence_tolerance, 1e-6) + size_penalty / max(cfg.split_size_tolerance, 1e-6)


def split_dataframe(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    groups = connected_component_groups(df, cfg)
    all_idx = np.arange(len(df))
    candidates: List[Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    for attempt in range(cfg.split_retries):
        rs = cfg.seed + attempt
        outer = GroupShuffleSplit(n_splits=1, test_size=cfg.test_fraction, random_state=rs)
        train_val_idx, test_idx = next(outer.split(all_idx, groups=groups))

        target_val = cfg.val_fraction / (1.0 - cfg.test_fraction)
        inner = GroupShuffleSplit(n_splits=1, test_size=target_val, random_state=rs + 100_000)
        inner_groups = groups[train_val_idx]
        train_sub, val_sub = next(inner.split(train_val_idx, groups=inner_groups))

        train = df.iloc[train_val_idx[train_sub]].reset_index(drop=True)
        val = df.iloc[train_val_idx[val_sub]].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        if not split_is_valid(train, val, test, cfg):
            continue
        score = split_score(train, val, test, cfg)
        candidates.append((score, train, val, test))

    if not candidates:
        raise RuntimeError(
            "Could not construct a valid leakage-safe train/validation/test split. "
            "This usually means the requested cold-start axis has too few independent "
            "groups or the binary endpoint is too sparse. Inspect entity cardinalities."
        )

    candidates.sort(key=lambda x: x[0])
    _, train, val, test = candidates[0]

    # Strong post-hoc assertions.
    for entity_col in [cfg.isolate_id_column, "observation_hash", *split_entity_columns(cfg)]:
        if not entity_col:
            continue
        for left, right, label in [
            (train, val, "train/val"),
            (train, test, "train/test"),
            (val, test, "val/test"),
        ]:
            overlap = split_overlap(left, right, [entity_col])[entity_col]
            if overlap:
                raise AssertionError(f"Leakage detected in {label} for {entity_col}: {overlap}")

    report = {
        "strategy": cfg.split_strategy,
        "total_rows": len(df),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_binary_n": int(train["binary_valid"].sum()),
        "val_binary_n": int(val["binary_valid"].sum()),
        "test_binary_n": int(test["binary_valid"].sum()),
        "train_positive_prevalence": prevalence(train),
        "val_positive_prevalence": prevalence(val),
        "test_positive_prevalence": prevalence(test),
        "group_count": int(len(np.unique(groups))),
        "cold_start_columns": split_entity_columns(cfg),
        "strict_joint_cold_available": bool(cfg.split_strategy == "joint_cold"),
        "integrity_columns_always_connected": [cfg.isolate_id_column, "observation_hash"] + ([cfg.assay_id_column] if cfg.assay_id_column in df.columns else []),
        "candidate_count": len(candidates),
        "selected_candidate_score": float(candidates[0][0]),
        "overlap_checks": {
            c: {
                "train_val": split_overlap(train, val, [c])[c],
                "train_test": split_overlap(train, test, [c])[c],
                "val_test": split_overlap(val, test, [c])[c],
            }
            for c in [cfg.isolate_id_column, "observation_hash", *split_entity_columns(cfg)]
            if c
        },
    }
    return train, val, test, report


# =============================================================================
# GRAPH ENCODER
# =============================================================================

class SMILESGATEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 96, heads: int = 4, dropout: float = 0.10, edge_dim: int = 14) -> None:
        super().__init__()
        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            residual=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
            residual=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.node_projection = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.gat1(batch.x, batch.edge_index, batch.edge_attr)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.gat2(h, batch.edge_index, batch.edge_attr)
        h = self.norm2(h)
        h = F.gelu(h)

        atom_tokens = self.node_projection(h)
        dense_atoms, atom_mask = to_dense_batch(atom_tokens, batch.batch)
        mean_pool = global_mean_pool(h, batch.batch)
        max_pool = global_max_pool(h, batch.batch)
        graph_embed = self.graph_projection(torch.cat([mean_pool, max_pool], dim=-1))
        return dense_atoms, atom_mask, graph_embed


# =============================================================================
# MODEL
# =============================================================================

class FusionBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class EpiResNetUnifiedV5(nn.Module):
    """ESM-2/LoRA + edge-aware GAT + bidirectional co-attention + dual task heads."""

    def __init__(self, cfg: Config, atom_feature_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        base_esm = EsmModel.from_pretrained(cfg.model_name)
        embed_dim = int(base_esm.config.hidden_size)
        self.esm_max_positions = int(getattr(base_esm.config, "max_position_embeddings", 1024))
        self.embed_dim = embed_dim

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["query", "key", "value"],
            bias="none",
        )
        self.esm = get_peft_model(base_esm, peft_config)
        self.drug_encoder = SMILESGATEncoder(
            in_dim=atom_feature_dim,
            out_dim=embed_dim,
            hidden_dim=cfg.atom_hidden,
            heads=cfg.atom_heads,
            dropout=cfg.graph_dropout,
            edge_dim=edge_feature_dim,
        )

        self.protein_to_drug_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=cfg.fusion_heads,
            dropout=cfg.fusion_dropout, batch_first=True,
        )
        self.drug_to_protein_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=cfg.fusion_heads,
            dropout=cfg.fusion_dropout, batch_first=True,
        )
        self.protein_norm = nn.LayerNorm(embed_dim)
        self.drug_norm = nn.LayerNorm(embed_dim)
        self.protein_ffn = FusionBlock(embed_dim, cfg.fusion_ffn_dim, cfg.fusion_dropout)
        self.drug_ffn = FusionBlock(embed_dim, cfg.fusion_ffn_dim, cfg.fusion_dropout)

        combined_dim = embed_dim * 3  # pooled protein, pooled drug atoms, graph global
        self.classifier_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(256, 1),
        )
        # Mean + log std of latent log2 MIC distribution.
        self.mic_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(256, 2),
        )

        self._assert_lora_configuration()

    def _assert_lora_configuration(self) -> None:
        trainable_base = []
        for name, p in self.esm.named_parameters():
            if p.requires_grad and "lora_" not in name.lower():
                trainable_base.append(name)
        if trainable_base:
            raise RuntimeError(
                "Unexpected trainable ESM parameters outside LoRA adapters: "
                + ", ".join(trainable_base[:10])
            )

    def parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_percent": float(100.0 * trainable / max(total, 1)),
        }

    def forward(
        self,
        esm_inputs: Mapping[str, torch.Tensor],
        protein_pool_mask: torch.Tensor,
        graphs: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        esm_outputs = self.esm(**esm_inputs)
        seq_reps = esm_outputs.last_hidden_state
        atom_tokens, atom_mask, graph_embed = self.drug_encoder(graphs)

        p2d, _ = self.protein_to_drug_attn(
            query=seq_reps,
            key=atom_tokens,
            value=atom_tokens,
            key_padding_mask=~atom_mask,
        )
        fused_protein = self.protein_norm(seq_reps + p2d)
        fused_protein = self.protein_ffn(fused_protein)

        d2p, _ = self.drug_to_protein_attn(
            query=atom_tokens,
            key=seq_reps,
            value=seq_reps,
            key_padding_mask=~protein_pool_mask,
        )
        fused_drug = self.drug_norm(atom_tokens + d2p)
        fused_drug = self.drug_ffn(fused_drug)

        p_mask = protein_pool_mask.to(fused_protein.dtype).unsqueeze(-1)
        pooled_protein = (fused_protein * p_mask).sum(dim=1) / p_mask.sum(dim=1).clamp_min(1.0)

        a_mask = atom_mask.to(fused_drug.dtype).unsqueeze(-1)
        pooled_drug = (fused_drug * a_mask).sum(dim=1) / a_mask.sum(dim=1).clamp_min(1.0)

        combined = torch.cat([pooled_protein, pooled_drug, graph_embed], dim=-1)
        binary_logits = self.classifier_head(combined).squeeze(-1)
        mic_raw = self.mic_head(combined)
        mic_mu = mic_raw[:, 0]
        mic_log_sigma = mic_raw[:, 1].clamp(min=-5.0, max=4.0)
        return binary_logits, mic_mu, mic_log_sigma


# =============================================================================
# LOSSES
# =============================================================================

class BinaryFocalLoss(nn.Module):
    """Numerically stable focal loss using BCE-with-logits."""

    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros(())
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal = (1.0 - p_t).pow(self.gamma)
        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal * bce).mean()


class IntervalCensoredNormalNLL(nn.Module):
    """Stable normal negative log-likelihood for exact/censored log2 MIC data."""

    def __init__(self, sigma_floor: float = 0.05) -> None:
        super().__init__()
        self.sigma_floor = float(sigma_floor)

    @staticmethod
    def _logdiffexp(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
        """Stable log(exp(log_a) - exp(log_b)), requiring log_a >= log_b."""
        out = torch.full_like(log_a, -torch.inf)
        valid = log_a > log_b
        if valid.any():
            delta = (log_b[valid] - log_a[valid]).clamp_max(-1e-7)
            out[valid] = log_a[valid] + torch.log1p(-torch.exp(delta))
        return out

    @classmethod
    def _log_interval_prob(cls, z_lower: torch.Tensor, z_upper: torch.Tensor) -> torch.Tensor:
        log_cdf_l = torch.special.log_ndtr(z_lower)
        log_cdf_u = torch.special.log_ndtr(z_upper)
        return cls._logdiffexp(log_cdf_u, log_cdf_l)

    def forward(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        censor_type: Sequence[str],
    ) -> torch.Tensor:
        if mu.numel() == 0:
            return mu.new_zeros(())

        mu = mu.float()
        log_sigma = log_sigma.float()
        lower = lower.float()
        upper = upper.float()
        sigma = F.softplus(log_sigma) + self.sigma_floor
        z_lower = (lower - mu) / sigma
        z_upper = (upper - mu) / sigma
        losses = torch.empty_like(mu)

        exact = torch.tensor([c == "exact" for c in censor_type], device=mu.device, dtype=torch.bool)
        left = torch.tensor([c == "left" for c in censor_type], device=mu.device, dtype=torch.bool)
        right = torch.tensor([c == "right" for c in censor_type], device=mu.device, dtype=torch.bool)
        interval = torch.tensor([c == "interval" for c in censor_type], device=mu.device, dtype=torch.bool)

        if exact.any():
            y = lower[exact]
            s = sigma[exact]
            losses[exact] = 0.5 * ((y - mu[exact]) / s).pow(2) + torch.log(s) + 0.5 * math.log(2.0 * math.pi)
        if left.any():
            losses[left] = -torch.special.log_ndtr(z_upper[left]).clamp_min(-80.0)
        if right.any():
            losses[right] = -torch.special.log_ndtr(-z_lower[right]).clamp_min(-80.0)
        if interval.any():
            log_prob = self._log_interval_prob(z_lower[interval], z_upper[interval])
            losses[interval] = -log_prob.clamp_min(-80.0)

        known = exact | left | right | interval
        if not bool(known.all()):
            raise ValueError(f"Unsupported censor types: {set(censor_type)}")
        return losses.mean()


# =============================================================================
# DATASET / COLLATOR
# =============================================================================

class AMRDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, graphs: Optional[List[Data]] = None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.graphs = graphs if graphs is not None else [smiles_to_graph(s) for s in self.df["smiles"]]
        if len(self.graphs) != len(self.df):
            raise ValueError("Number of graphs does not match dataframe rows")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "sequence": row["protein_sequence"],
            "graph": self.graphs[idx],
            "label": float(row["phenotype"]),
            "binary_valid": bool(row["binary_valid"]),
            "mic_lower": float(row["mic_log2_lower"]),
            "mic_upper": float(row["mic_log2_upper"]),
            "censor_type": row["mic_censor_type"],
        }


class AMRCollator:
    def __init__(self, tokenizer, max_seq_len: int, truncate: bool, overflow_strategy: str = "error") -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate
        self.overflow_strategy = overflow_strategy
        if overflow_strategy not in {"error", "n_terminal", "center"}:
            raise ValueError("overflow_strategy must be error, n_terminal, or center")
        if truncate and overflow_strategy == "error":
            raise ValueError("truncate=True requires an explicit overflow_strategy")

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Batch, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        sequences = [item["sequence"] for item in batch]
        graphs = [item["graph"] for item in batch]
        labels = torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32)
        binary_valid = torch.tensor([bool(item["binary_valid"]) for item in batch], dtype=torch.bool)
        lower = torch.tensor([float(item["mic_lower"]) for item in batch], dtype=torch.float32)
        upper = torch.tensor([float(item["mic_upper"]) for item in batch], dtype=torch.float32)
        censor_types = [str(item["censor_type"]) for item in batch]

        if self.truncate:
            encoded_sequences = []
            max_residues = self.max_seq_len - 2
            for seq in sequences:
                if len(seq) + 2 <= self.max_seq_len:
                    encoded_sequences.append(seq)
                    continue
                if self.overflow_strategy == "n_terminal":
                    encoded_sequences.append(seq[:max_residues])
                elif self.overflow_strategy == "center":
                    start = max(0, (len(seq) - max_residues) // 2)
                    encoded_sequences.append(seq[start:start + max_residues])
                else:
                    raise ValueError(f"Unsupported overflow strategy: {self.overflow_strategy}")
            sequences = encoded_sequences

        tok = self.tokenizer(
            sequences,
            padding=True,
            truncation=False,
            max_length=self.max_seq_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special_mask = tok.pop("special_tokens_mask")
        protein_pool_mask = tok["attention_mask"].bool() & ~special_mask.bool()
        if not protein_pool_mask.any(dim=1).all():
            raise RuntimeError("At least one protein sequence has no poolable residue tokens.")

        graph_batch = Batch.from_data_list(graphs)
        return tok, protein_pool_mask, graph_batch, labels, binary_valid, lower, upper, censor_types


# =============================================================================
# OPTIMIZATION / METRICS
# =============================================================================

def derive_focal_alpha(train_df: pd.DataFrame) -> float:
    valid = train_df.loc[train_df["binary_valid"], "phenotype"].astype(int)
    n_pos = int((valid == 1).sum())
    n_neg = int((valid == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training split must contain both binary classes.")
    alpha = n_neg / (n_pos + n_neg)
    return float(np.clip(alpha, 0.50, 0.95))


def parameter_groups(model: nn.Module, cfg: Config) -> List[Dict[str, Any]]:
    """Heterogeneous LR groups with AdamW no-decay handling for norms/biases."""
    buckets: Dict[str, Dict[str, List[nn.Parameter]]] = {
        "lora": {"decay": [], "nodecay": []},
        "graph": {"decay": [], "nodecay": []},
        "head": {"decay": [], "nodecay": []},
        "other": {"decay": [], "nodecay": []},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "lora_" in lname:
            bucket = "lora"
        elif "drug_encoder" in lname:
            bucket = "graph"
        elif any(x in lname for x in ("classifier_head", "mic_head", "protein_to_drug_attn", "drug_to_protein_attn", "protein_ffn", "drug_ffn")):
            bucket = "head"
        else:
            bucket = "other"
        no_decay = (p.ndim == 1) or lname.endswith(".bias") or "norm" in lname or "layernorm" in lname
        buckets[bucket]["nodecay" if no_decay else "decay"].append(p)

    lr_map = {"lora": cfg.lr_lora, "graph": cfg.lr_graph, "head": cfg.lr_head, "other": cfg.lr_head}
    out: List[Dict[str, Any]] = []
    for bucket, parts in buckets.items():
        if parts["decay"]:
            out.append({"params": parts["decay"], "lr": lr_map[bucket], "weight_decay": cfg.weight_decay})
        if parts["nodecay"]:
            out.append({"params": parts["nodecay"], "lr": lr_map[bucket], "weight_decay": 0.0})
    if not out:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return out


def choose_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "mcc") -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("Threshold selection requires both binary classes in validation data.")
    _, _, thresholds = precision_recall_curve(y_true, probs)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], thresholds)))
    scored: List[Tuple[float, float]] = []
    for t in candidates:
        pred = (probs >= t).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_true, pred)
        elif metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if np.isfinite(score):
            scored.append((float(score), float(t)))
    if not scored:
        raise RuntimeError("Could not score any validation threshold.")
    scored.sort(key=lambda x: (-x[0], abs(x[1] - 0.5), x[1]))
    return scored[0][1]


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if not mask.any():
            continue
        conf = float(probs[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def fit_temperature(y_true: np.ndarray, probs: np.ndarray, max_iter: int = 100) -> float:
    """Fit a single validation-set temperature for probability calibration."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 1.0
    p = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float64)
    target = torch.tensor(y_true.astype(np.float64), dtype=torch.float64)
    log_t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / t, target)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.05, 20.0).detach().cpu())


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(p / (1.0 - p))
    calibrated = 1.0 / (1.0 + np.exp(-logits / max(float(temperature), 1e-6)))
    return calibrated.astype(np.float64)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "mcc", "brier", "ece"]}
    pred = (probs >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
    }


def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            tok, pool_mask, graphs, y, valid, *_ = batch
            valid = valid.to(device)
            if not bool(valid.any()):
                continue
            tok = {k: v.to(device) for k, v in tok.items()}
            pool_mask = pool_mask.to(device)
            graphs = graphs.to(device)
            logits, _, _ = model(tok, pool_mask, graphs)
            probs = torch.sigmoid(logits)
            all_probs.append(probs[valid].float().cpu().numpy())
            all_y.append(y[valid].cpu().numpy())
    if not all_y:
        raise RuntimeError("No valid binary observations available for evaluation")
    y_true = np.concatenate(all_y)
    probs = np.concatenate(all_probs)
    return {"y_true": y_true, "probs": probs}


def evaluate_mic(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    exact_y: List[np.ndarray] = []
    exact_pred: List[np.ndarray] = []
    interval_hit: List[bool] = []
    interval_nll: List[float] = []
    nll_loss = IntervalCensoredNormalNLL(sigma_floor=0.05)

    with torch.no_grad():
        for batch in loader:
            tok, pool_mask, graphs, _, _, lower, upper, censor = batch
            tok = {k: v.to(device) for k, v in tok.items()}
            pool_mask = pool_mask.to(device)
            graphs = graphs.to(device)
            _, mu, log_sigma = model(tok, pool_mask, graphs)

            pred = mu.float().cpu().numpy()
            lo = lower.numpy()
            hi = upper.numpy()
            interval_hit.extend(((pred >= lo) & (pred <= hi)).tolist())

            batch_nll = nll_loss(mu, log_sigma, lower.to(device), upper.to(device), censor)
            interval_nll.append(float(batch_nll.detach().cpu()))

            mask_exact = np.array([c == "exact" for c in censor], dtype=bool)
            if mask_exact.any():
                exact_y.append(lo[mask_exact])
                exact_pred.append(pred[mask_exact])

    metrics: Dict[str, float] = {
        "interval_point_hit_rate": float(np.mean(interval_hit)) if interval_hit else float("nan"),
        "mean_interval_nll": float(np.mean(interval_nll)) if interval_nll else float("nan"),
    }
    if exact_y:
        y = np.concatenate(exact_y)
        p = np.concatenate(exact_pred)
        metrics.update({
            "exact_mae_log2_mic": float(mean_absolute_error(y, p)),
            "exact_rmse_log2_mic": float(math.sqrt(mean_squared_error(y, p))),
            "exact_r2_log2_mic": float(r2_score(y, p)) if len(y) >= 2 else float("nan"),
        })
    else:
        metrics.update({
            "exact_mae_log2_mic": float("nan"),
            "exact_rmse_log2_mic": float("nan"),
            "exact_r2_log2_mic": float("nan"),
        })
    return metrics


# =============================================================================
# TRAINING
# =============================================================================

def amp_settings(device: torch.device, mode: str) -> Tuple[bool, Optional[torch.dtype], bool]:
    mode = mode.lower()
    if device.type != "cuda" or mode == "off":
        return False, None, False
    if mode == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif mode == "bf16":
        dtype = torch.bfloat16
        if not torch.cuda.is_bf16_supported():
            warnings.warn("Requested bf16 but GPU does not advertise bf16 support; using fp16.")
            dtype = torch.float16
    elif mode == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("mixed_precision must be off, bf16, fp16, or auto")
    scaler_enabled = dtype == torch.float16
    return True, dtype, scaler_enabled


def move_batch(batch: Tuple[Any, ...], device: torch.device):
    tok, pool_mask, graphs, y, valid, lower, upper, censor = batch
    tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
    return (
        tok,
        pool_mask.to(device, non_blocking=True),
        graphs.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        valid.to(device, non_blocking=True),
        lower.to(device, non_blocking=True),
        upper.to(device, non_blocking=True),
        censor,
    )


def run_training(csv_path: str, cfg: Config) -> None:
    seed_everything(cfg.seed, cfg.deterministic)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.sequence_overflow_strategy not in {"error", "n_terminal", "center"}:
        raise ValueError("Invalid sequence_overflow_strategy")
    if cfg.truncate_long_sequences and cfg.sequence_overflow_strategy == "error":
        raise ValueError("Truncation requires an explicit sequence_overflow_strategy")
    csv_file = Path(csv_path).resolve()
    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/8] Device: {device}")
    raw_df = pd.read_csv(csv_file)
    print(f"[2/8] Loaded {len(raw_df):,} raw rows from {csv_file.name}")
    df = prepare_assay_dataframe(raw_df, cfg)
    print(f"[3/8] Prepared {len(df):,} unique assay observations")
    print(f"      Binary-valid observations: {int(df['binary_valid'].sum()):,}")
    print(f"      Binary-positive prevalence: {prevalence(df):.3f}")

    train_df, val_df, test_df, split_report = split_dataframe(df, cfg)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    write_json(output_dir / "split_report.json", split_report)
    print(f"[4/8] Split = {cfg.split_strategy}: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model_max = getattr(tokenizer, "model_max_length", 1024)
    max_seq_len = cfg.max_seq_len or min(model_max, 1024)
    if max_seq_len > model_max and model_max < 10_000:
        raise ValueError(f"max_seq_len={max_seq_len} exceeds tokenizer model_max_length={model_max}")

    # ESM-2 positional limits include special tokens; reject silent residue loss.
    lengths = [len(s) + 2 for s in df["protein_sequence"]]
    overlong = [i for i, x in enumerate(lengths) if x > max_seq_len]
    if overlong and not cfg.truncate_long_sequences:
        raise ValueError(
            f"{len(overlong)} proteins exceed max_seq_len={max_seq_len}. "
            "Default policy is to fail rather than silently discard residues. "
            "Enable explicit truncation only with a justified sequence-overflow strategy."
        )
    if overlong and cfg.sequence_overflow_strategy == "error":
        raise ValueError("Overlong sequences require sequence_overflow_strategy != 'error'.")

    atom_dim, edge_dim = feature_dimensions()
    print(f"[5/8] Molecular feature dimensions: atom={atom_dim}, edge={edge_dim}")

    # Cache graphs once for each split.
    train_graphs = [smiles_to_graph(s) for s in train_df["smiles"]]
    val_graphs = [smiles_to_graph(s) for s in val_df["smiles"]]
    test_graphs = [smiles_to_graph(s) for s in test_df["smiles"]]

    train_ds = AMRDataset(train_df, train_graphs)
    val_ds = AMRDataset(val_df, val_graphs)
    test_ds = AMRDataset(test_df, test_graphs)
    collator = AMRCollator(tokenizer, max_seq_len=max_seq_len, truncate=cfg.truncate_long_sequences, overflow_strategy=cfg.sequence_overflow_strategy)

    common_loader = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker,
        collate_fn=collator,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader)

    model = EpiResNetUnifiedV5(cfg, atom_feature_dim=atom_dim, edge_feature_dim=edge_dim).to(device)
    summary = model.parameter_summary()
    print(f"      Parameters: {summary['total_parameters']:,}; trainable={summary['trainable_parameters']:,} ({summary['trainable_percent']:.2f}%)")

    alpha = derive_focal_alpha(train_df)
    focal_loss = BinaryFocalLoss(alpha=alpha, gamma=2.0)
    mic_loss = IntervalCensoredNormalNLL(cfg.mic_sigma_floor)

    optimizer = AdamW(parameter_groups(model, cfg))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp, amp_dtype, scaler_enabled = amp_settings(device, cfg.mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled) if device.type == "cuda" else None

    manifest = {
        "config": asdict(cfg),
        "input_csv": str(csv_file),
        "input_sha256": file_sha256(csv_file),
        "source_rows": len(raw_df),
        "prepared_rows": len(df),
        "deduplicated_rows": int(df.attrs.get("deduplicated_rows", 0)),
        "split_report": split_report,
        "model_summary": summary,
        "focal_alpha": alpha,
        "atom_feature_dim": atom_dim,
        "edge_feature_dim": edge_dim,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "torch_geometric": package_version("torch-geometric"),
        "rdkit": package_version("rdkit"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "device": str(device),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    best_metric = -np.inf
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    print("[6/8] Training")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            tok, pool_mask, graphs, y, valid, lower, upper, censor = move_batch(batch, device)
            if use_amp:
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                ctx = torch.autocast(device_type=device.type, enabled=False)

            with ctx:
                logits, mu, log_sigma = model(tok, pool_mask, graphs)
                if bool(valid.any()):
                    loss_bin = focal_loss(logits[valid], y[valid])
                else:
                    loss_bin = logits.new_zeros(())
                loss_mic = mic_loss(mu, log_sigma, lower, upper, censor)
                total_loss = loss_bin + cfg.mic_loss_weight * loss_mic
                total_loss = total_loss / cfg.grad_accum_steps

            if scaler_enabled and scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if batch_idx % cfg.grad_accum_steps == 0 or batch_idx == len(train_loader):
                if scaler_enabled and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler_enabled and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(total_loss.detach().cpu()) * cfg.grad_accum_steps
            batches += 1

        val_binary = evaluate_binary(model, val_loader, device)
        temperature = fit_temperature(val_binary["y_true"], val_binary["probs"], cfg.calibration_max_iter) if cfg.calibrate_probabilities else 1.0
        val_probs_cal = apply_temperature(val_binary["probs"], temperature)
        val_threshold = choose_threshold(val_binary["y_true"], val_probs_cal, cfg.threshold_metric)
        val_clf = classification_metrics(val_binary["y_true"], val_probs_cal, val_threshold)
        val_mic = evaluate_mic(model, val_loader, device)
        metric = val_clf["auprc"]
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "val_threshold": val_threshold,
            "val_auprc": metric,
            **{f"val_{k}": v for k, v in val_clf.items()},
            **{f"val_{k}": v for k, v in val_mic.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss={row['train_loss']:.4f} | "
            f"val AUPRC={row['val_auprc']:.4f} | AUROC={row['val_auroc']:.4f} | "
            f"MCC={row['val_mcc']:.4f} | thr={val_threshold:.3f}"
        )

        if metric > best_metric + cfg.min_delta:
            best_metric = metric
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "focal_alpha": alpha,
                "best_val_auprc": best_metric,
                "best_val_threshold": val_threshold,
                "probability_temperature": float(temperature),
                "epoch": epoch,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    print(f"[7/8] Loading best checkpoint (epoch {best_epoch})")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_binary = evaluate_binary(model, val_loader, device)
    threshold = float(checkpoint["best_val_threshold"])
    temperature = float(checkpoint.get("probability_temperature", 1.0))
    test_binary = evaluate_binary(model, test_loader, device)
    test_probs_cal = apply_temperature(test_binary["probs"], temperature)
    test_clf = classification_metrics(test_binary["y_true"], test_probs_cal, threshold)
    test_mic = evaluate_mic(model, test_loader, device)

    final_metrics = {
        "selection": {
            "metric": "validation_auprc",
            "best_validation_auprc": float(best_metric),
            "best_epoch": int(best_epoch),
            "threshold_metric": cfg.threshold_metric,
            "selected_threshold_on_validation": float(threshold),
            "probability_temperature": float(temperature),
            "threshold_source": "checkpoint-selected-validation-threshold",
        },
        "test_classification": test_clf,
        "test_mic": test_mic,
    }
    write_json(output_dir / "final_metrics.json", final_metrics)

    predictions = test_df.copy().reset_index(drop=True)
    # Preserve all test observations; binary predictions are only populated where
    # the clinical breakpoint-derived target is valid.
    binary_probs = np.full(len(predictions), np.nan, dtype=float)
    binary_pred = np.full(len(predictions), np.nan, dtype=float)
    valid_positions = np.flatnonzero(predictions["binary_valid"].to_numpy(dtype=bool))
    if len(valid_positions) != len(test_binary["probs"]):
        raise AssertionError("Prediction rows do not match binary-valid test observations.")
    binary_probs[valid_positions] = test_probs_cal
    binary_pred[valid_positions] = (test_probs_cal >= threshold).astype(int)
    predictions["probability_resistant"] = binary_probs
    predictions["prediction_resistant"] = binary_pred
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print("[8/8] Final test evaluation")
    print(json.dumps(final_metrics, indent=2))
    print(f"Artifacts written to: {output_dir.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production-oriented ESM-2 + GAT multimodal AMR pipeline with assay-aware MIC targets and leakage-safe cold-start splits."
    )
    parser.add_argument("--csv", required=True, help="Input assay CSV")
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--split-strategy", choices=Config.allowed_split_strategies, default=Config.split_strategy)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=Config.grad_accum_steps)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--mixed-precision", choices=["off", "bf16", "fp16", "auto"], default=Config.mixed_precision)
    parser.add_argument("--truncate-long-sequences", action="store_true")
    parser.add_argument("--sequence-overflow-strategy", choices=["error", "n_terminal", "center"], default=Config.sequence_overflow_strategy)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--validate-only", action="store_true", help="Validate/prepare data and construct the split without training")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = Config(
        split_strategy=args.split_strategy,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        truncate_long_sequences=args.truncate_long_sequences,
        sequence_overflow_strategy=args.sequence_overflow_strategy,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    seed_everything(cfg.seed, cfg.deterministic)

    if args.validate_only:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        raw = pd.read_csv(args.csv)
        prepared = prepare_assay_dataframe(raw, cfg)
        train, val, test, report = split_dataframe(prepared, cfg)
        train.to_csv(output / "train_split.csv", index=False)
        val.to_csv(output / "val_split.csv", index=False)
        test.to_csv(output / "test_split.csv", index=False)
        write_json(output / "split_report.json", report)
        print(json.dumps(report, indent=2))
        print("Validation-only run completed successfully.")
        return

    run_training(args.csv, cfg)


if __name__ == "__main__":
    main()
