"""Click CLI for WhyMove."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from whymove.classifier.labels import ALL_LABELS
from whymove.container import AppConfig
from whymove.engine.factory import create_engine
from whymove.models import ExplanationResult, LabeledIntent, MoveInput


@click.group()
@click.option(
    "--engine-depth", default=20, show_default=True, type=int,
    help="Stockfish search depth.",
)
@click.option(
    "--model-path", default="models/intent_classifier.joblib", show_default=True,
    help="Path to trained classifier model.",
)
@click.option(
    "--stockfish-path", default=None,
    help="Path to Stockfish binary (auto-detected if not set).",
)
@click.pass_context
def cli(ctx: click.Context, engine_depth: int, model_path: str, stockfish_path: str | None) -> None:
    """WhyMove — explain the strategic intent behind chess moves."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = AppConfig(
        engine_depth=engine_depth,
        model_path=Path(model_path),
        stockfish_path=stockfish_path,
    )


@cli.command("analyze")
@click.argument("move_uci")
@click.option("--fen", default=None, help="Position in FEN notation.")
@click.option("--pgn", "pgn_file", default=None, type=click.Path(exists=True),
              help="Path to PGN file (uses final position).")
@click.option("--pgn-text", default=None, help="PGN string (inline).")
@click.option("--top-k", default=5, show_default=True, help="Number of intent labels to show.")
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON.")
@click.pass_context
def analyze_cmd(
    ctx: click.Context,
    move_uci: str,
    fen: str | None,
    pgn_file: str | None,
    pgn_text: str | None,
    top_k: int,
    output_json: bool,
) -> None:
    """Analyze a chess move and explain its intent.

    \b
    Examples:
      whymove analyze --fen "rnbq1rk1/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQ - 2 6" d4d5
      whymove analyze --pgn game.pgn e2e4
    """
    config: AppConfig = ctx.obj["config"]

    # Resolve PGN string
    pgn_str: str | None = pgn_text
    if pgn_file and pgn_str is None:
        pgn_str = Path(pgn_file).read_text()

    if fen is None and pgn_str is None:
        raise click.UsageError("Provide either --fen or --pgn / --pgn-text.")

    move_input = MoveInput(
        fen=fen,
        pgn=pgn_str,
        move_uci=move_uci,
        top_k_labels=top_k,
        engine_depth=config.engine_depth,
    )

    try:
        from whymove.classifier.model import IntentClassifier
        from whymove.explainer.claude_client import ClaudeExplainer
        from whymove.pipeline import AnalysisPipeline

        with create_engine(config.engine_type, path=config.stockfish_path, depth=config.engine_depth) as engine:
            clf = IntentClassifier.load(config.model_path)
            explainer = ClaudeExplainer(model=config.claude_model)
            pipeline = AnalysisPipeline(engine, clf, explainer, engine_depth=config.engine_depth)
            result = pipeline.analyze(move_input)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output_json:
        click.echo(result.model_dump_json(indent=2))
    else:
        _print_result(result)


@cli.command("train")
@click.option("--features", "features_path", required=True, type=click.Path(exists=True),
              help="Path to features.parquet file.")
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True),
              help="Path to labels JSONL file.")
@click.option("--output", default="models/intent_classifier.joblib", show_default=True,
              help="Output path for trained model.")
@click.option("--threshold", default=0.3, show_default=True, type=float,
              help="Confidence threshold for intent labels.")
@click.option("--test-size", default=0.2, show_default=True, type=float,
              help="Fraction of data for evaluation.")
def train_cmd(
    features_path: str,
    labels_path: str,
    output: str,
    threshold: float,
    test_size: float,
) -> None:
    """Train the intent classifier from extracted features and labels."""
    from whymove.classifier.training import load_training_data, train_model
    import json as json_mod

    click.echo(f"Loading data from {features_path} + {labels_path}...")
    X, y = load_training_data(Path(features_path), Path(labels_path))
    click.echo(f"  {len(X)} samples, {len(ALL_LABELS)} labels")

    click.echo("Training classifier...")
    clf, metrics = train_model(X, y, threshold=threshold, test_size=test_size)

    output_path = Path(output)
    clf.save(output_path)
    click.echo(f"Model saved to {output_path}")

    # Save metrics
    metrics_path = output_path.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json_mod.dump(metrics, f, indent=2)

    click.echo(f"\nResults:")
    click.echo(f"  Macro F1: {metrics['macro_f1']:.3f}")
    click.echo(f"  Micro F1: {metrics['micro_f1']:.3f}")
    click.echo(f"  Metrics saved to {metrics_path}")


@cli.command("extract-features")
@click.argument("pgn_file", type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path(),
              help="Output path for features.parquet.")
@click.option("--depth", default=15, show_default=True, help="Engine analysis depth.")
@click.option("--max-games", default=None, type=int, help="Maximum number of games to process.")
@click.option("--stockfish-path", default=None, help="Path to Stockfish binary.")
def extract_features_cmd(
    pgn_file: str,
    output: str,
    depth: int,
    max_games: int | None,
    stockfish_path: str | None,
) -> None:
    """Extract features from all moves in a PGN file.

    \b
    Example:
      whymove extract-features data/raw/games.pgn --output data/processed/features.parquet
    """
    from scripts.extract_features import process_pgn_file

    process_pgn_file(
        pgn_path=Path(pgn_file),
        output_path=Path(output),
        depth=depth,
        max_games=max_games,
        stockfish_path=stockfish_path,
    )


@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Enable hot reload (development mode).")
def serve_cmd(host: str, port: int, reload: bool) -> None:
    """Start the WhyMove REST API server."""
    import uvicorn
    click.echo(f"Starting WhyMove API server at http://{host}:{port}")
    uvicorn.run(
        "whymove.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


def _print_result(result: ExplanationResult) -> None:
    """Print analysis result in a human-friendly format."""
    click.echo()
    click.echo(f"Move: {result.move_san}")
    click.echo(f"Position: {result.fen_before}")
    click.echo()
    click.echo("Intent labels:")
    for intent in result.intents:
        bar = "█" * int(intent.confidence * 10)
        click.echo(f"  {intent.label.value:<30} {intent.confidence:.0%}  {bar}")
    click.echo()
    click.echo("Explanation:")
    click.echo(f"  {result.explanation}")
    click.echo()
    fs = result.feature_summary
    click.echo(
        f"Features: eval_delta={fs['eval_delta_cp']:+d}cp  "
        f"material_delta={fs['material_delta']:+d}cp  "
        f"check={fs['is_check']}  capture={fs['is_capture']}"
    )
