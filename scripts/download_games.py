"""Stream-filter a Lichess .pgn.zst database file and extract games by rating.

No API key, no username, no account needed.

Usage:
    python scripts/download_games.py \\
        --input ~/Downloads/lichess_db_standard_rated_2024-01.pgn.zst \\
        --output data/raw/games.pgn \\
        --min-elo 2000 \\
        --max-games 10000
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import chess.pgn
import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def stream_filter_games(
    zst_path: Path,
    output_path: Path,
    min_elo: int = 2000,
    max_games: int = 10000,
    min_moves: int = 10,
) -> None:
    try:
        import zstandard as zstd
    except ImportError:
        click.echo("Install zstandard first: pip install zstandard", err=True)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    scanned = 0

    click.echo(f"Streaming {zst_path.name} ...")
    click.echo(f"Filtering: both players >= {min_elo} ELo, >= {min_moves} moves, {max_games} games target")

    dctx = zstd.ZstdDecompressor()

    with open(zst_path, "rb") as f, open(output_path, "w") as out:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

            while count < max_games:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    click.echo("Reached end of file.")
                    break

                scanned += 1

                # Filter by ELo
                try:
                    white_elo = int(game.headers.get("WhiteElo", "0") or "0")
                    black_elo = int(game.headers.get("BlackElo", "0") or "0")
                except ValueError:
                    continue

                if white_elo < min_elo or black_elo < min_elo:
                    continue

                # Filter by game length (short games have less interesting moves)
                moves = list(game.mainline_moves())
                if len(moves) < min_moves:
                    continue

                out.write(str(game) + "\n\n")
                count += 1

                if count % 500 == 0:
                    click.echo(f"  {count}/{max_games} games collected (scanned {scanned} total)...")

    click.echo(f"\nDone. Collected {count} games (scanned {scanned}).")
    click.echo(f"Saved to {output_path}")


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Path to .pgn.zst file from database.lichess.org")
@click.option("--output", default="data/raw/games.pgn", show_default=True,
              help="Output .pgn file path")
@click.option("--min-elo", default=2000, show_default=True,
              help="Minimum ELo for both players")
@click.option("--max-games", default=10000, show_default=True,
              help="Stop after collecting this many games")
@click.option("--min-moves", default=10, show_default=True,
              help="Skip games shorter than this many moves")
def main(input_path: str, output: str, min_elo: int, max_games: int, min_moves: int) -> None:
    stream_filter_games(
        zst_path=Path(input_path),
        output_path=Path(output),
        min_elo=min_elo,
        max_games=max_games,
        min_moves=min_moves,
    )


if __name__ == "__main__":
    main()
