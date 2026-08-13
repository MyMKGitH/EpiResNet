from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any, Dict

import streamlit as st

# -----------------------------------------------------------------------------
# Project path setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# The production pipeline's default output directory is artifacts/epiresnet_v5.
ARTIFACT_DIR = BASE_DIR / "artifacts" / "epiresnet_v5"
MODEL_PATH = ARTIFACT_DIR / "best_model.pt"
MANIFEST_PATH = ARTIFACT_DIR / "run_manifest.json"
METRICS_PATH = ARTIFACT_DIR / "final_metrics.json"

# -----------------------------------------------------------------------------
# Import the actual production-v5 model API.
# -----------------------------------------------------------------------------
PIPELINE_IMPORT_ERROR: Exception | None = None

try:
    import torch
    import torch.nn.functional as F
    from rdkit import Chem
    from transformers import AutoTokenizer
    from torch_geometric.data import Batch

    from epiresnet_production_v5 import (
        Config,
        EpiResNetUnifiedV5,
        feature_dimensions,
        smiles_to_graph,
        canonicalize_smiles,
    )
except Exception as exc:  # noqa: BLE001
    PIPELINE_IMPORT_ERROR = exc


st.set_page_config(
    page_title="EpiResNet v5 Engine",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 EpiResNet v5: Real-Time Multimodal AMR Predictor")
st.markdown(
    "Production inference engine using **ESM-2 + LoRA**, "
    "an **edge-aware GAT**, and **bidirectional protein↔drug co-attention**."
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def config_from_checkpoint(raw_config: Any) -> Config:
    """Reconstruct Config from the dict saved by the production training code."""
    if isinstance(raw_config, Config):
        return raw_config

    if not isinstance(raw_config, dict):
        return Config()

    # Keep compatibility if a future checkpoint contains extra metadata keys.
    valid_fields = set(Config.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw_config.items() if k in valid_fields}
    return Config(**filtered)


def read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sequence_for_inference(sequence: str, tokenizer, cfg: Config) -> str:
    """Apply the model's explicit production sequence-overflow policy."""
    sequence = sequence.replace(" ", "").strip().upper()
    if not sequence:
        raise ValueError("Protein sequence is empty.")

    max_seq_len = cfg.max_seq_len
    if max_seq_len is None:
        model_max = getattr(tokenizer, "model_max_length", 1024)
        max_seq_len = min(model_max, 1024)

    # The production collator reserves two special tokens.
    max_residues = max_seq_len - 2
    if len(sequence) + 2 <= max_seq_len:
        return sequence

    if not cfg.truncate_long_sequences:
        raise ValueError(
            f"Protein sequence has {len(sequence)} residues, but the active "
            f"configuration permits at most {max_residues} residues "
            f"({max_seq_len} tokens including special tokens). "
            "The production model is configured to reject silent truncation."
        )

    strategy = cfg.sequence_overflow_strategy
    if strategy == "n_terminal":
        return sequence[:max_residues]
    if strategy == "center":
        start = max(0, (len(sequence) - max_residues) // 2)
        return sequence[start : start + max_residues]

    raise ValueError(
        "Sequence exceeds the configured maximum and the active "
        "sequence_overflow_strategy is 'error'."
    )


@st.cache_resource(show_spinner=False)
def load_production_model():
    """Load the exact production-v5 checkpoint and its associated metadata."""
    if PIPELINE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Could not import the production-v5 pipeline. "
            f"Original error: {PIPELINE_IMPORT_ERROR!r}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production checkpoint not found. Expected:\n"
            f"{MODEL_PATH}\n\n"
            "Train epiresnet_production_v5.py first, or copy its "
            "best_model.pt into artifacts/epiresnet_v5/."
        )

    # Production v5 was saved with torch.save(checkpoint, ...), and the
    # checkpoint contains config, state_dict, validation threshold and temperature.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    cfg = config_from_checkpoint(checkpoint.get("config", {}))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    atom_dim, edge_dim = feature_dimensions()
    model = EpiResNetUnifiedV5(
        cfg,
        atom_feature_dim=atom_dim,
        edge_feature_dim=edge_dim,
    ).to(device)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch. Missing={missing}, unexpected={unexpected}"
        )

    model.eval()

    threshold = float(checkpoint.get("best_val_threshold", 0.5))
    temperature = float(checkpoint.get("probability_temperature", 1.0))

    manifest = read_json(MANIFEST_PATH)
    final_metrics = read_json(METRICS_PATH)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "cfg": cfg,
        "threshold": threshold,
        "temperature": temperature,
        "atom_dim": atom_dim,
        "edge_dim": edge_dim,
        "checkpoint": checkpoint,
        "manifest": manifest,
        "final_metrics": final_metrics,
    }


def calibrated_probability(raw_prob: float, temperature: float) -> float:
    """Apply the same temperature calibration used at production evaluation."""
    p = min(max(float(raw_prob), 1e-7), 1.0 - 1e-7)
    logit = torch.tensor(p / (1.0 - p)).log()
    calibrated = torch.sigmoid(logit / max(float(temperature), 1e-6))
    return float(calibrated.item())


# -----------------------------------------------------------------------------
# Load model — this is the normal app path, not diagnostics-only behavior.
# -----------------------------------------------------------------------------
try:
    assets = load_production_model()
    model = assets["model"]
    tokenizer = assets["tokenizer"]
    device = assets["device"]
    cfg = assets["cfg"]
    threshold = assets["threshold"]
    temperature = assets["temperature"]
    manifest = assets["manifest"]
    final_metrics = assets["final_metrics"]

    st.success(
        f"✅ Production checkpoint loaded successfully — running on **{device}**."
    )

except Exception as exc:  # noqa: BLE001
    model = None
    tokenizer = None
    device = None
    cfg = None
    threshold = 0.5
    temperature = 1.0
    manifest = None
    final_metrics = None

    st.error("❌ Model could not be loaded.")
    st.exception(exc)
    st.info(
        "The app is configured for the production-v5 checkpoint at "
        f"`{MODEL_PATH}`. This is the only expected checkpoint location "
        "for this Streamlit app."
    )


# -----------------------------------------------------------------------------
# Main inference UI
# -----------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📥 Input Query")

    protein_seq = st.text_area(
        "Target Protein Sequence",
        value="MKKTAIAIAVALAGFATVAQAAPKDNTWYTGAKL",
        height=180,
        help="Amino-acid sequence for the target protein.",
    ).strip()

    smiles = st.text_input(
        "Antibiotic SMILES",
        value="CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        help="RDKit-compatible antibiotic SMILES string.",
    ).strip()

    run_btn = st.button(
        "🚀 Run Real Inference",
        type="primary",
        disabled=model is None,
        use_container_width=True,
    )

with right:
    st.subheader("📊 Output Predictions")

    if model is None:
        st.warning("Load the production checkpoint to enable inference.")

    elif run_btn:
        try:
            # Validate and canonicalize exactly through the production pipeline.
            canonical_smiles, mol = canonicalize_smiles(smiles)
            sequence = sequence_for_inference(protein_seq, tokenizer, cfg)

            # Single-example graph batch: EpiResNetUnifiedV5 expects a PyG Batch.
            graph = smiles_to_graph(canonical_smiles)
            graph_batch = Batch.from_data_list([graph]).to(device)

            # Tokenization follows AMRCollator: no automatic residue truncation,
            # plus explicit removal of ESM special tokens from the pooling mask.
            max_seq_len = cfg.max_seq_len
            if max_seq_len is None:
                model_max = getattr(tokenizer, "model_max_length", 1024)
                max_seq_len = min(model_max, 1024)

            encoded = tokenizer(
                [sequence],
                padding=True,
                truncation=False,
                max_length=max_seq_len,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )

            special_mask = encoded.pop("special_tokens_mask")
            protein_pool_mask = (
                encoded["attention_mask"].bool()
                & ~special_mask.bool()
            ).to(device)

            esm_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            with st.spinner(
                "Running ESM-2 + LoRA, molecular GAT and bidirectional co-attention..."
            ):
                with torch.no_grad():
                    logits, mic_mu, mic_log_sigma = model(
                        esm_inputs,
                        protein_pool_mask,
                        graph_batch,
                    )

            # ---------------------------------------------------------
            # Binary AMR prediction
            # ---------------------------------------------------------
            raw_prob = float(torch.sigmoid(logits).item())
            prob_resistant = calibrated_probability(raw_prob, temperature)
            predicted_resistant = prob_resistant >= threshold
            phenotype = "Resistant (R)" if predicted_resistant else "Susceptible (S)"
            confidence = (
                prob_resistant
                if predicted_resistant
                else 1.0 - prob_resistant
            )

            # ---------------------------------------------------------
            # MIC prediction
            # Production model returns mean + log-sigma in latent log2(MIC)
            # space. Its training loss uses sigma = softplus(log_sigma)
            # + sigma_floor, so inference must do the same.
            # ---------------------------------------------------------
            log2_mic = float(mic_mu.item())
            log_sigma = float(mic_log_sigma.item())
            sigma_log2 = float(
                (F.softplus(mic_log_sigma) + cfg.mic_sigma_floor).item()
            )

            mic_mg_l = float(2.0 ** log2_mic)
            lower_mic = float(2.0 ** (log2_mic - 1.96 * sigma_log2))
            upper_mic = float(2.0 ** (log2_mic + 1.96 * sigma_log2))

            st.success("✅ Inference completed successfully.")

            r1, r2 = st.columns(2)
            with r1:
                st.metric(
                    "Predicted Phenotype",
                    phenotype,
                    f"{confidence * 100:.1f}% model confidence",
                    delta_color="inverse" if predicted_resistant else "normal",
                )
            with r2:
                st.metric(
                    "Predicted MIC",
                    f"{mic_mg_l:.3f} mg/L",
                    f"log₂(MIC) = {log2_mic:.2f}",
                )

            st.write(
                f"**Approx. 95% MIC interval:** {lower_mic:.3f} – {upper_mic:.3f} mg/L"
            )

            with st.expander("🔬 Prediction Details", expanded=True):
                st.write(f"**Canonical SMILES:** `{canonical_smiles}`")
                st.write(f"**Protein length used:** {len(sequence)} residues")
                st.write(f"**Raw resistance probability:** {raw_prob:.6f}")
                st.write(f"**Calibrated resistance probability:** {prob_resistant:.6f}")
                st.write(f"**Frozen validation threshold:** {threshold:.6f}")
                st.write(f"**Probability temperature:** {temperature:.6f}")
                st.write(f"**Latent MIC mean, log₂ scale:** {log2_mic:.6f}")
                st.write(f"**Latent MIC σ, log₂ scale:** {sigma_log2:.6f}")
                st.write(f"**Inference device:** {device}")

            st.caption(
                "The threshold and probability temperature are taken from the production "
                "checkpoint; they are not re-estimated during inference."
            )

        except Exception as exc:  # noqa: BLE001
            st.error("❌ Inference failed.")
            st.exception(exc)


# -----------------------------------------------------------------------------
# Model lineage / metadata
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🛡️ Active Production Model")

if model is not None:
    info1, info2, info3 = st.columns(3)
    with info1:
        st.metric("Model", "EpiResNetUnifiedV5")
    with info2:
        st.metric("Device", str(device))
    with info3:
        st.metric("Threshold", f"{threshold:.4f}")

    if manifest:
        with st.expander("View run manifest"):
            st.json(manifest)

    if final_metrics:
        with st.expander("View final test metrics"):
            st.json(final_metrics)

else:
    st.warning(
        f"Expected production checkpoint: `{MODEL_PATH}`"
    )
