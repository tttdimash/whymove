"""Extract features from PGN games and save to Parquet.

Usage:
    python scripts/extract_features.py \\
        --pgn data/raw/games.pgn \\
        --output data/processed/features.parquet \\
        --depth 15 \\
        --max-games 10000
"""

from __future__ import annotations

import io
import sys
import uuid
from pathlib import Path

import chess
import chess.pgn
import click
import pandas as pd

# Add src to path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whymove.engine.factory import create_engine
from whymove.features.extractor import FeatureExtractor
from whymove.features.vectorizer import FEATURE_NAMES, features_to_vector


def process_pgn_file(
    pgn_path: Path,
    output_path: Path,
    depth: int = 15,
    max_games: int | None = None,
    stockfish_path: str | None = None,
) -> None:
    """Extract features from all moves in a PGN file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    game_count = 0
    total_moves = 0
    SAVE_EVERY = 100  # save to disk every 100 games

    with create_engine("stockfish", path=stockfish_path, depth=depth) as engine:
        extractor = FeatureExtractor(engine)

        with open(pgn_path, encoding="latin-1") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                if max_games and game_count >= max_games:
                    break

                game_id = game.headers.get("Site", "") or str(uuid.uuid4())[:8]
                board = game.board()

                for move_num, node in enumerate(game.mainline()):
                    move = node.move
                    fen = board.fen()

                    try:
                        features = extractor.extract(fen, move.uci(), depth)
                        vec = features_to_vector(features)
                        row = dict(zip(FEATURE_NAMES, vec.tolist()))
                        row["move_id"] = f"{game_id}_{move_num}"
                        row["game_id"] = game_id
                        row["move_num"] = move_num
                        row["fen"] = fen
                        row["move_uci"] = move.uci()
                        row["move_san"] = board.san(move)
                        rows.append(row)
                        total_moves += 1
                    except Exception as e:
                        click.echo(f"  Warning: skipped move {move.uci()} in {game_id}: {e}", err=True)

                    board.push(move)

                game_count += 1

                # Save incrementally every SAVE_EVERY games
                if game_count % SAVE_EVERY == 0:
                    df = pd.DataFrame(rows)
                    df.to_parquet(output_path, index=False)
                    target = max_games or "?"
                    pct = int(game_count / max_games * 100) if max_games else "?"
                    click.echo(f"[{pct}%] {game_count}/{target} games | {total_moves:,} moves | saved ✓")

    # Final save
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    click.echo(f"\nDone. {game_count} games, {total_moves:,} moves saved to {output_path}")


@click.command()
@click.argument("pgn_file", type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path(), help="Output .parquet path")
@click.option("--depth", default=15, show_default=True)
@click.option("--max-games", default=None, type=int)
@click.option("--stockfish-path", default=None)
def main(pgn_file: str, output: str, depth: int, max_games: int | None, stockfish_path: str | None) -> None:
    process_pgn_file(
        pgn_path=Path(pgn_file),
        output_path=Path(output),
        depth=depth,
        max_games=max_games,
        stockfish_path=stockfish_path,
    )


if __name__ == "__main__":
    main()
