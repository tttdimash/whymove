"""Train the intent classifier from features + labels.

Usage:
    python scripts/train_model.py \\
        --features data/processed/features.parquet \\
        --labels data/labels/claude_labels.jsonl \\
        --output models/intent_classifier.joblib
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whymove.classifier.labels import ALL_LABELS
from whymove.classifier.training import load_training_data, train_model


@click.command()
@click.option("--features", required=True, type=click.Path(exists=True))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True))
@click.option("--output", default="models/intent_classifier.joblib", show_default=True)
@click.option("--threshold", default=0.3, show_default=True, type=float)
@click.option("--test-size", default=0.2, show_default=True, type=float)
def main(
    features: str,
    labels_path: str,
    output: str,
    threshold: float,
    test_size: float,
) -> None:
    """Train and save the intent classifier."""
    click.echo(f"Loading data...")
    X, y = load_training_data(Path(features), Path(labels_path))
    click.echo(f"  {len(X)} samples loaded")

    click.echo("Training MultiOutputClassifier(GradientBoosting)...")
    clf, metrics = train_model(X, y, threshold=threshold, test_size=test_size)

    output_path = Path(output)
    clf.save(output_path)
    click.echo(f"Model saved → {output_path}")

    metrics_path = output_path.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"Metrics saved → {metrics_path}")

    click.echo(f"\n{'Label':<35} {'F1':>6}")
    click.echo("-" * 43)
    for label, f1 in sorted(metrics["per_label_f1"].items(), key=lambda x: -x[1]):
        click.echo(f"  {label:<33} {f1:.3f}")
    click.echo("-" * 43)
    click.echo(f"  {'Macro F1':<33} {metrics['macro_f1']:.3f}")
    click.echo(f"  {'Micro F1':<33} {metrics['micro_f1']:.3f}")


if __name__ == "__main__":
    main()
