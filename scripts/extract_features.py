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

    with create_engine("stockfish", path=stockfish_path, depth=depth) as engine:
        extractor = FeatureExtractor(engine)

        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                if max_games and game_count >= max_games:
                    break

                white = game.headers.get("White", "?").replace(" ", "_")[:20]
                black = game.headers.get("Black", "?").replace(" ", "_")[:20]
                date = game.headers.get("Date", "????.??.??").replace(".", "")[:8]
                round_ = game.headers.get("Round", "?").replace(".", "_")
                game_id = f"{white}_vs_{black}_{date}_r{round_}"
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
                    except Exception as e:
                        click.echo(f"  Warning: skipped move {move.uci()} in {game_id}: {e}", err=True)

                    board.push(move)

                game_count += 1
                if game_count % 100 == 0:
                    click.echo(f"Processed {game_count} games, {len(rows)} moves...")

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    click.echo(f"Saved {len(df)} feature rows to {output_path}")


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
