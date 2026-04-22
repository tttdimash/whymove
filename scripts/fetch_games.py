"""Fetch diverse training games from Lichess API.

Strategy:
- Pull from players across a rating range (1800-2400) for position diversity
- Mix time controls (classical > rapid > blitz) for move quality
- Sample from many different players to avoid one player's opening repertoire dominating
- No file download needed — streams directly from Lichess API

No API key required. Free and public.

Usage:
    python scripts/fetch_games.py \
        --output data/raw/games.pgn \
        --target-games 10000 \
        --min-elo 1800 \
        --max-elo 2400
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess.pgn
import click
import requests
import io

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LICHESS_BASE = "https://lichess.org"

# Time controls in order of move quality (classical = more thinking time = better intent signal)
PERF_TYPES = ["classical", "rapid", "blitz"]

# How many games to fetch per player — keep small so we sample many different players
# rather than getting 1000 games from the same person (same opening repertoire)
GAMES_PER_PLAYER = 50


def fetch_players_in_range(min_elo: int, max_elo: int, count: int = 300) -> list[str]:
    """
    Get a list of Lichess usernames in a given rating range.
    Uses the Lichess player search API.
    """
    usernames: list[str] = []

    # Lichess has a leaderboard per perf type - we use this to seed our player list
    # then filter by rating range
    for perf in PERF_TYPES:
        if len(usernames) >= count:
            break
        url = f"{LICHESS_BASE}/api/player/top/200/{perf}"
        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
            r.raise_for_status()
            for user in r.json().get("users", []):
                name = user.get("username", "")
                rating = user.get("perfs", {}).get(perf, {}).get("rating", 0)
                if name and min_elo <= rating <= max_elo and name not in usernames:
                    usernames.append(name)
        except Exception as e:
            click.echo(f"  Warning: leaderboard fetch failed for {perf}: {e}", err=True)
        time.sleep(1)

    click.echo(f"Found {len(usernames)} players in {min_elo}–{max_elo} range")
    return usernames


def count_games_in_pgn(pgn_text: str) -> int:
    return pgn_text.count("[Event ")


def count_moves_in_pgn(pgn_text: str) -> int:
    """Count total half-moves across all games — this is what matters for training."""
    total = 0
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        total += len(list(game.mainline_moves()))
    return total


def fetch_games_for_user(
    username: str,
    max_games: int = GAMES_PER_PLAYER,
    perf_types: str = "classical,rapid",
    min_rating: int = 1800,
    max_rating: int = 2400,
) -> str:
    """Fetch PGN games for one user, filtered by rating range."""
    url = f"{LICHESS_BASE}/api/games/user/{username}"
    params = {
        "max": max_games,
        "rated": "true",
        "perfType": perf_types,
        "clocks": "false",
        "evals": "false",
        "opening": "false",
        # Filter to games played around our target rating range
        "ratingRange": f"{min_rating}-{max_rating}",
    }
    headers = {"Accept": "application/x-chess-pgn"}

    try:
        r = requests.get(url, params=params, headers=headers, stream=True, timeout=30)
        if r.status_code == 429:
            click.echo("  Rate limited — waiting 60s...", err=True)
            time.sleep(60)
            r = requests.get(url, params=params, headers=headers, stream=True, timeout=30)
        r.raise_for_status()
        return r.text
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {r.status_code}: {e}") from e


def fetch_games(
    output_path: Path,
    target_games: int = 10000,
    min_elo: int = 1800,
    max_elo: int = 2400,
    games_per_player: int = GAMES_PER_PLAYER,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Get a diverse pool of players in the rating range
    click.echo(f"Finding players rated {min_elo}–{max_elo}...")
    usernames = fetch_players_in_range(min_elo, max_elo, count=300)

    if not usernames:
        click.echo("No players found. Check your rating range.", err=True)
        sys.exit(1)

    # Step 2: Fetch games from each player, stop when we have enough
    total_games = 0
    total_moves = 0
    players_used = 0

    click.echo(f"\nFetching up to {games_per_player} games per player...")
    click.echo(f"Target: {target_games} games from as many different players as possible\n")

    with open(output_path, "w") as out:
        for i, username in enumerate(usernames):
            if total_games >= target_games:
                break

            click.echo(f"  [{i+1}/{len(usernames)}] {username}...", nl=False)

            try:
                pgn_text = fetch_games_for_user(
                    username,
                    max_games=games_per_player,
                    min_rating=min_elo,
                    max_rating=max_elo,
                )
                game_count = count_games_in_pgn(pgn_text)
                if game_count == 0:
                    click.echo(" no games in range, skipping")
                    continue

                out.write(pgn_text)
                if not pgn_text.endswith("\n\n"):
                    out.write("\n\n")

                total_games += game_count
                players_used += 1
                click.echo(f" +{game_count} games (total: {total_games})")

            except Exception as e:
                click.echo(f" failed: {e}", err=True)

            # Lichess rate limit: ~20 requests/min for anonymous users
            time.sleep(3)

    click.echo(f"\nDone.")
    click.echo(f"  Games collected : {total_games}")
    click.echo(f"  Players sampled : {players_used}")
    click.echo(f"  Saved to        : {output_path}")
    click.echo(f"\nNext step:")
    click.echo(f"  python scripts/extract_features.py {output_path} --output data/processed/features.parquet")


@click.command()
@click.option("--output", default="data/raw/games.pgn", show_default=True,
              help="Output .pgn file path")
@click.option("--target-games", default=10000, show_default=True,
              help="Stop after collecting this many games")
@click.option("--min-elo", default=1800, show_default=True,
              help="Minimum player rating")
@click.option("--max-elo", default=2400, show_default=True,
              help="Maximum player rating")
@click.option("--games-per-player", default=GAMES_PER_PLAYER, show_default=True,
              help="Max games per player (keep low for diversity)")
def main(
    output: str,
    target_games: int,
    min_elo: int,
    max_elo: int,
    games_per_player: int,
) -> None:
    """Fetch diverse chess games from Lichess API for training data."""
    fetch_games(
        output_path=Path(output),
        target_games=target_games,
        min_elo=min_elo,
        max_elo=max_elo,
        games_per_player=games_per_player,
    )


if __name__ == "__main__":
    main()
