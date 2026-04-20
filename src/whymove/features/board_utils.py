"""Pure python-chess helpers for computing position metrics."""

from __future__ import annotations

import chess


# ── Material ──────────────────────────────────────────────────────────────────

PIECE_VALUES_CP: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def get_material_value(board: chess.Board) -> int:
    """Return white material minus black material in centipawns."""
    total = 0
    for piece_type, value in PIECE_VALUES_CP.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


# ── Mobility ──────────────────────────────────────────────────────────────────

def get_piece_mobility(board: chess.Board, square: chess.Square) -> int:
    """Count legal moves for the piece on the given square."""
    piece = board.piece_at(square)
    if piece is None:
        return 0
    return sum(1 for m in board.legal_moves if m.from_square == square)


def get_total_mobility(board: chess.Board, color: chess.Color) -> int:
    """Count total legal moves for all pieces of color."""
    # Push a null move if it's the opponent's turn so we can count color's moves.
    if board.turn == color:
        return sum(1 for _ in board.legal_moves)
    # Temporarily flip the board (be careful: only valid in non-check positions)
    try:
        board.push(chess.Move.null())
        count = sum(1 for _ in board.legal_moves)
        board.pop()
        return count
    except Exception:
        return 0


# ── Geometry ──────────────────────────────────────────────────────────────────

def chebyshev_distance(sq1: chess.Square, sq2: chess.Square) -> int:
    """Chebyshev (king-move) distance between two squares."""
    r1, f1 = chess.square_rank(sq1), chess.square_file(sq1)
    r2, f2 = chess.square_rank(sq2), chess.square_file(sq2)
    return max(abs(r1 - r2), abs(f1 - f2))


# ── King Safety ───────────────────────────────────────────────────────────────

_KING_ZONE_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
    (-2, 0),  (2, 0),  (0, -2), (0, 2),
]


def _king_zone_squares(king_sq: chess.Square) -> list[chess.Square]:
    rank, file = chess.square_rank(king_sq), chess.square_file(king_sq)
    squares = [king_sq]
    for dr, df in _KING_ZONE_OFFSETS:
        r, f = rank + dr, file + df
        if 0 <= r <= 7 and 0 <= f <= 7:
            squares.append(chess.square(f, r))
    return squares


def get_king_zone_attackers(board: chess.Board, king_color: chess.Color) -> int:
    """Count opponent pieces attacking squares in the king zone."""
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0
    attacker_color = not king_color
    zone = _king_zone_squares(king_sq)
    attackers: set[chess.Square] = set()
    for sq in zone:
        for attacker_sq in board.attackers(attacker_color, sq):
            attackers.add(attacker_sq)
    return len(attackers)


def get_open_files_near_king(board: chess.Board, king_color: chess.Color) -> int:
    """Count open/half-open files within 1 file of the king."""
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    count = 0
    for f in range(max(0, king_file - 1), min(8, king_file + 2)):
        white_pawns = any(
            chess.square_file(sq) == f
            for sq in board.pieces(chess.PAWN, chess.WHITE)
        )
        black_pawns = any(
            chess.square_file(sq) == f
            for sq in board.pieces(chess.PAWN, chess.BLACK)
        )
        if not white_pawns or not black_pawns:
            count += 1
    return count


# ── Pawn Structure ────────────────────────────────────────────────────────────

def count_doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count pawns that share a file with a friendly pawn (each extra counts)."""
    files: dict[int, int] = {}
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        files[f] = files.get(f, 0) + 1
    return sum(n - 1 for n in files.values() if n > 1)


def count_isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count pawns with no friendly pawns on adjacent files."""
    pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
    return sum(
        1 for f in pawn_files
        if (f - 1) not in pawn_files and (f + 1) not in pawn_files
    )


def count_passed_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count pawns with no opposing pawns on the same or adjacent files ahead of them."""
    opponent = not color
    opp_pawn_sqs = board.pieces(chess.PAWN, opponent)
    count = 0
    for sq in board.pieces(chess.PAWN, color):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        direction = 1 if color == chess.WHITE else -1
        blocked = False
        for opp_sq in opp_pawn_sqs:
            opp_rank = chess.square_rank(opp_sq)
            opp_file = chess.square_file(opp_sq)
            if abs(opp_file - file) <= 1:
                # Check if opponent pawn is ahead
                if color == chess.WHITE and opp_rank > rank:
                    blocked = True
                    break
                if color == chess.BLACK and opp_rank < rank:
                    blocked = True
                    break
        if not blocked:
            count += 1
    return count


def count_pawn_islands(board: chess.Board, color: chess.Color) -> int:
    """Count pawn islands (groups of pawns on consecutive files)."""
    pawn_files = sorted({chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)})
    if not pawn_files:
        return 0
    islands = 1
    for i in range(1, len(pawn_files)):
        if pawn_files[i] != pawn_files[i - 1] + 1:
            islands += 1
    return islands


# ── Square Control ────────────────────────────────────────────────────────────

_CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}
_KEY_SQUARES = _CENTER_SQUARES | {
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.D6,
    chess.E3, chess.E6,
    chess.F3, chess.F4, chess.F5, chess.F6,
}


def count_center_squares_controlled(board: chess.Board, color: chess.Color) -> int:
    """Count center squares (d4/d5/e4/e5) attacked by color."""
    return sum(1 for sq in _CENTER_SQUARES if board.is_attacked_by(color, sq))


def count_key_squares_controlled(board: chess.Board, color: chess.Color) -> int:
    """Count extended-center squares attacked by color."""
    return sum(1 for sq in _KEY_SQUARES if board.is_attacked_by(color, sq))


# ── FEN utilities ─────────────────────────────────────────────────────────────

def fen_after_move(fen: str, move_uci: str) -> str:
    """Return the FEN after applying a UCI move to the given position."""
    board = chess.Board(fen)
    board.push_uci(move_uci)
    return board.fen()


def normalize_eval_to_white(score_cp: int, turn: chess.Color) -> int:
    """Convert a side-to-move-relative centipawn score to white-relative."""
    return score_cp if turn == chess.WHITE else -score_cp
