from pathlib import Path
import json
import torch
from dataclasses import asdict

# Import requirements from your production script
from epiresnet_production_v5 import (
    Config,
    EpiResNetUnifiedV3,
    feature_dimensions,
)

def generate_mock_artifacts():
    # 1. Ensure artifact output directory exists
    output_dir = Path("artifacts/epiresnet_v5")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Target directory: {output_dir.resolve()}")

    # 2. Instantiate default configuration and compute dimensions
    cfg = Config()
    atom_dim, edge_dim = feature_dimensions()

    print("🧬 Initializing EpiResNetUnifiedV3 architecture (downloading ESM-2 backbone if needed)...")
    model = EpiResNetUnifiedV3(
        cfg,
        atom_feature_dim=atom_dim,
        edge_feature_dim=edge_dim,
    )

    # 3. Create model checkpoint dictionary matching production format
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "focal_alpha": 0.5,
        "best_val_auprc": 0.85,
        "best_val_threshold": 0.5,
        "probability_temperature": 1.0,
        "epoch": 1,
    }

    model_path = output_dir / "best_model.pt"
    torch.save(checkpoint, model_path)
    print(f"✅ Saved dummy checkpoint -> {model_path}")

    # 4. Create mock run_manifest.json
    manifest = {
        "config": asdict(cfg),
        "device": "cpu",
        "status": "mock_generated_for_testing",
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✅ Saved manifest -> {manifest_path}")

    # 5. Create mock final_metrics.json
    metrics = {
        "selection": {
            "selected_threshold_on_validation": 0.5,
            "probability_temperature": 1.0,
        },
        "test_classification": {
            "auroc": 0.912,
            "auprc": 0.884,
            "accuracy": 0.865,
            "mcc": 0.723,
        },
        "test_mic": {
            "exact_mae_log2_mic": 0.45,
            "exact_rmse_log2_mic": 0.62,
        },
    }
    metrics_path = output_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"✅ Saved metrics -> {metrics_path}")

    print("\n🎉 Artifact generation complete! You can now launch Streamlit.")

if __name__ == "__main__":
    generate_mock_artifacts()
