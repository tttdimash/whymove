"""Generate training labels using Claude for a sampled set of positions.

Usage:
    python scripts/generate_labels.py \\
        --features data/processed/features.parquet \\
        --output data/labels/claude_labels.jsonl \\
        --sample 10000
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whymove.classifier.labels import ALL_LABELS
from whymove.explainer.claude_client import ClaudeExplainer
from whymove.features.vectorizer import FEATURE_NAMES
from whymove.models import (
    KingSafetyFeatures,
    PawnStructureFeatures,
    PieceInfo,
    PositionFeatures,
    TacticalFlags,
)


def _row_to_features(row: dict) -> PositionFeatures:
    """Reconstruct a PositionFeatures from a feature dict row."""
    return PositionFeatures(
        eval_before_cp=int(row.get("eval_before_cp", 0)),
        eval_after_cp=int(row.get("eval_after_cp", 0)),
        eval_delta_cp=int(row["eval_delta_cp"]),
        material_before=int(row.get("material_before", 0)),
        material_after=int(row.get("material_after", 0)),
        material_delta=int(row["material_delta"]),
        moved_piece_mobility_before=int(row["moved_piece_mobility_before"]),
        moved_piece_mobility_after=int(row["moved_piece_mobility_after"]),
        total_white_mobility_delta=int(row["total_white_mobility_delta"]),
        total_black_mobility_delta=int(row["total_black_mobility_delta"]),
        piece=PieceInfo(
            piece_type=1,  # Simplified: pawn as placeholder
            color=bool(row.get("piece_color", True)),
            from_square=0,
            to_square=int(row.get("destination_rank", 0) * 8 + row.get("destination_file", 0)),
        ),
        distance_moved=int(row["distance_moved"]),
        destination_rank=int(row["destination_rank"]),
        destination_file=int(row["destination_file"]),
        king_safety=KingSafetyFeatures(
            white_king_zone_attacks_delta=int(row["white_king_zone_attacks_delta"]),
            black_king_zone_attacks_delta=int(row["black_king_zone_attacks_delta"]),
            white_open_files_near_king_delta=int(row["white_open_files_near_king_delta"]),
            black_open_files_near_king_delta=int(row["black_open_files_near_king_delta"]),
        ),
        pawn_structure=PawnStructureFeatures(
            doubled_pawns_delta=int(row["doubled_pawns_delta"]),
            isolated_pawns_delta=int(row["isolated_pawns_delta"]),
            passed_pawns_delta=int(row["passed_pawns_delta"]),
            pawn_islands_delta=int(row["pawn_islands_delta"]),
        ),
        center_control_delta=int(row["center_control_delta"]),
        key_square_control_delta=int(row["key_square_control_delta"]),
        tactical=TacticalFlags(
            is_fork=bool(row["is_fork"]),
            is_pin=bool(row["is_pin"]),
            is_skewer=bool(row["is_skewer"]),
            is_discovered_attack=bool(row["is_discovered_attack"]),
            is_double_attack=bool(row["is_double_attack"]),
            is_double_check=bool(row["is_double_check"]),
            is_mating_threat=bool(row["is_mating_threat"]),
            is_capture=bool(row["is_capture"]),
            is_promotion=bool(row["is_promotion"]),
            is_check=bool(row["is_check"]),
            is_zwischenzug=bool(row["is_zwischenzug"]),
            is_x_ray=bool(row["is_x_ray"]),
            is_overloading=bool(row["is_overloading"]),
            is_sacrifice=bool(row["is_sacrifice"]),
            is_deflection=bool(row["is_deflection"]),
            is_interposition=bool(row["is_interposition"]),
        ),
    )


def generate_labels_for_batch(
    features_path: Path,
    output_path: Path,
    sample_size: int = 10000,
    delay_seconds: float = 0.5,
    resume: bool = True,
) -> None:
    """Generate labels for a sampled subset of positions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    # Resume: skip already-labeled move_ids
    labeled_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    labeled_ids.add(json.loads(line)["move_id"])
                except Exception:
                    pass
        click.echo(f"Resuming: {len(labeled_ids)} already labeled")

    explainer = ClaudeExplainer(model="claude-haiku-4-5-20251001", max_tokens=50)
    all_labels = [l.value for l in ALL_LABELS]

    with open(output_path, "a") as out_f:
        for i, (_, row) in enumerate(df.iterrows()):
            move_id = str(row["move_id"])
            if move_id in labeled_ids:
                continue

            fen = str(row["fen"])
            move_san = str(row.get("move_san", "?"))

            try:
                features = _row_to_features(row.to_dict())
                labels = explainer.generate_labels(fen, move_san, features)
            except Exception as e:
                click.echo(f"  Warning: failed for {move_id}: {e}", err=True)
                labels = []

            record = {
                "move_id": move_id,
                "labels": labels,
                "fen": fen,
                "move_uci": str(row.get("move_uci", "")),
                "move_san": move_san,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (i + 1) % 100 == 0:
                click.echo(f"Labeled {i + 1} positions...")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    click.echo(f"Done. Labels saved to {output_path}")


@click.command()
@click.option("--features", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
@click.option("--sample", default=10000, show_default=True, type=int)
@click.option("--delay", default=0.5, show_default=True, type=float, help="Seconds between API calls")
@click.option("--no-resume", is_flag=True, help="Don't resume from existing output file")
def main(features: str, output: str, sample: int, delay: float, no_resume: bool) -> None:
    generate_labels_for_batch(
        features_path=Path(features),
        output_path=Path(output),
        sample_size=sample,
        delay_seconds=delay,
        resume=not no_resume,
    )


if __name__ == "__main__":
    main()
