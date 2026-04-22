"""Tactical motif detection using python-chess board analysis."""

from __future__ import annotations

import chess

from whymove.features.board_utils import PIECE_VALUES_CP


# ── Individual motif detectors ────────────────────────────────────────────────

def is_fork(board_before: chess.Board, move: chess.Move) -> bool:
    """After the move, the moved piece attacks 2+ opponent pieces of value >= knight."""
    board_after = board_before.copy()
    board_after.push(move)
    to_sq = move.to_square
    mover_color = board_before.turn
    opponent = not mover_color

    valuable_attacked: list[chess.Square] = []
    for sq in chess.SQUARES:
        piece = board_after.piece_at(sq)
        if piece and piece.color == opponent:
            # King is always a valid fork target even though its material value is 0
            if (
                piece.piece_type == chess.KING
                or PIECE_VALUES_CP.get(piece.piece_type, 0) >= PIECE_VALUES_CP[chess.KNIGHT]
            ):
                if board_after.is_attacked_by(mover_color, sq):
                    # Check that this specific piece on to_sq attacks sq
                    if sq in board_after.attacks(to_sq):
                        valuable_attacked.append(sq)

    return len(valuable_attacked) >= 2


def is_pin_created(board_after: chess.Board) -> bool:
    """Check if any opponent piece is pinned after the move."""
    # Use python-chess built-in pin detection
    for sq in chess.SQUARES:
        piece = board_after.piece_at(sq)
        if piece and piece.color == board_after.turn:
            if board_after.is_pinned(board_after.turn, sq):
                return True
    return False


def is_skewer(board_before: chess.Board, move: chess.Move) -> bool:
    """A sliding piece attacks a high-value piece; a piece behind it is also in the ray."""
    board_after = board_before.copy()
    board_after.push(move)
    to_sq = move.to_square
    moved_piece = board_after.piece_at(to_sq)
    if moved_piece is None:
        return False
    if moved_piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return False

    mover_color = board_before.turn
    opponent = not mover_color

    for sq in chess.SQUARES:
        target = board_after.piece_at(sq)
        if not target or target.color != opponent:
            continue
        if to_sq not in board_after.attackers(mover_color, sq):
            continue
        # Check if there's a piece behind the target along the same ray
        direction = _ray_direction(to_sq, sq)
        if direction is None:
            continue
        behind_sq = sq + direction
        while 0 <= behind_sq <= 63:
            behind_piece = board_after.piece_at(chess.Square(behind_sq))
            if behind_piece:
                if behind_piece.color == opponent:
                    if PIECE_VALUES_CP.get(target.piece_type, 0) > PIECE_VALUES_CP.get(
                        behind_piece.piece_type, 0
                    ):
                        return True
                break
            behind_sq += direction
    return False


def _ray_direction(from_sq: chess.Square, to_sq: chess.Square) -> int | None:
    """Return the square-index step along the ray from from_sq toward to_sq, or None."""
    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)
    tr, tf = chess.square_rank(to_sq), chess.square_file(to_sq)
    dr = tr - fr
    df = tf - ff

    if dr == 0 and df == 0:
        return None
    if dr == 0:
        return 1 if df > 0 else -1
    if df == 0:
        return 8 if dr > 0 else -8
    if abs(dr) == abs(df):
        return (8 if dr > 0 else -8) + (1 if df > 0 else -1)
    return None


def is_discovered_attack(board_before: chess.Board, move: chess.Move) -> bool:
    """Moving this piece reveals an attack by a piece behind it."""
    mover_color = board_before.turn
    opponent = not mover_color
    from_sq = move.from_square

    # Squares attacked by our pieces (excluding from_sq) before the move
    attackers_before: set[chess.Square] = set()
    for sq in chess.SQUARES:
        target = board_before.piece_at(sq)
        if not target or target.color != opponent:
            continue
        for att_sq in board_before.attackers(mover_color, sq):
            if att_sq != from_sq:
                attackers_before.add(sq)

    board_after = board_before.copy()
    board_after.push(move)
    to_sq = move.to_square

    # New squares attacked after the move (not by to_sq itself)
    for sq in chess.SQUARES:
        target = board_after.piece_at(sq)
        if not target or target.color == mover_color:
            continue
        if sq in attackers_before:
            continue
        for att_sq in board_after.attackers(mover_color, sq):
            if att_sq != to_sq:  # attack not from the moved piece itself
                return True
    return False


def is_double_check(board_after: chess.Board) -> bool:
    """Two or more pieces deliver check simultaneously."""
    return board_after.is_check() and len(list(board_after.checkers())) >= 2


def is_mating_threat(board_after: chess.Board, engine_evaluate: object | None = None) -> bool:
    """Heuristic: opponent has very few legal moves AND is in check, or only king moves."""
    if not board_after.is_check():
        return False
    legal = list(board_after.legal_moves)
    return len(legal) <= 3


def is_zwischenzug(board_before: chess.Board, move: chess.Move) -> bool:
    """Intermediate move: a recapture was available but this move ignores it."""
    if not board_before.is_check():
        return False  # Zwischenzug is most meaningful when expected recapture is skipped
    # A zwischenzug is an in-between move; a basic heuristic: the move is not a recapture
    # on the last captured square. This is a simplified heuristic.
    captured_sq = board_before.peek().to_square if board_before.move_stack else None
    if captured_sq is None:
        return False
    expected_recapture = chess.Move(move.from_square, captured_sq)
    return (
        expected_recapture in board_before.legal_moves
        and move.to_square != captured_sq
    )


def is_x_ray(board_before: chess.Board, move: chess.Move) -> bool:
    """A sliding piece attacks through one piece to attack another."""
    board_after = board_before.copy()
    board_after.push(move)
    to_sq = move.to_square
    moved_piece = board_after.piece_at(to_sq)
    if moved_piece is None:
        return False
    if moved_piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return False

    mover_color = board_before.turn
    opponent = not mover_color

    # Check if any opponent piece is shielded by a friendly piece but still in the ray
    for sq in chess.SQUARES:
        target = board_after.piece_at(sq)
        if not target or target.color != opponent:
            continue
        direction = _ray_direction(to_sq, sq)
        if direction is None:
            continue
        # Check for friendly pieces between to_sq and sq
        between_sq = to_sq + direction
        while between_sq != sq and 0 <= between_sq <= 63:
            between_piece = board_after.piece_at(chess.Square(between_sq))
            if between_piece and between_piece.color == mover_color:
                return True  # X-ray through own piece
            between_sq += direction
    return False


def is_overloading(board_before: chess.Board, move: chess.Move) -> bool:
    """An opponent piece is defending two things; this move threatens both."""
    board_after = board_before.copy()
    board_after.push(move)
    mover_color = board_before.turn
    opponent = not mover_color

    # Find opponent pieces that are defending 2+ squares we attack
    for def_sq in chess.SQUARES:
        defender = board_after.piece_at(def_sq)
        if not defender or defender.color != opponent:
            continue
        # What squares does this defender guard?
        defended = list(board_after.attacks(def_sq))
        # How many of those do we threaten?
        threatened = [
            sq for sq in defended
            if board_after.piece_at(sq) is not None
            and board_after.is_attacked_by(mover_color, sq)
        ]
        if len(threatened) >= 2:
            return True
    return False


def is_sacrifice(
    board_before: chess.Board,
    move: chess.Move,
    eval_delta_cp: int,
) -> bool:
    """A move that loses material (negative material delta) but gains positional value."""
    moved_piece = board_before.piece_at(move.from_square)
    if moved_piece is None:
        return False
    captured = board_before.piece_at(move.to_square)
    moved_value = PIECE_VALUES_CP.get(moved_piece.piece_type, 0)

    if captured:
        captured_value = PIECE_VALUES_CP.get(captured.piece_type, 0)
        material_loss = moved_value - captured_value
        # Sacrifice: we give up more than we take, but evaluation improves
        return material_loss > 50 and eval_delta_cp > 20
    else:
        # Pure positional sacrifice (no immediate capture)
        return False


def is_deflection(board_before: chess.Board, move: chess.Move) -> bool:
    """Move attacks or captures a piece that was defending a key square or piece."""
    board_after = board_before.copy()
    board_after.push(move)
    to_sq = move.to_square
    mover_color = board_before.turn
    opponent = not mover_color

    # Was the captured/attacked piece defending something important before?
    captured = board_before.piece_at(to_sq)
    if captured is None:
        return False

    # Check what the captured piece was defending
    previously_defended = [
        sq for sq in board_before.attacks(to_sq)
        if board_before.piece_at(sq) and board_before.piece_at(sq).color == opponent  # type: ignore[union-attr]
    ]
    # After removal, are those squares now undefended?
    for sq in previously_defended:
        if not board_after.is_attacked_by(opponent, sq):
            if board_after.is_attacked_by(mover_color, sq):
                return True
    return False


def is_interposition(board_before: chess.Board, move: chess.Move) -> bool:
    """Move blocks a check or an attack on the king."""
    if board_before.is_check():
        board_after = board_before.copy()
        board_after.push(move)
        return not board_after.is_check()
    return False


# ── Main aggregator ───────────────────────────────────────────────────────────

def compute_tactical_flags(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    eval_delta_cp: int,
) -> dict[str, bool]:
    """Compute all tactical flags for a move. Returns a dict matching TacticalFlags fields."""
    captured = board_before.piece_at(move.to_square)

    return {
        "is_fork": is_fork(board_before, move),
        "is_pin": is_pin_created(board_after),
        "is_skewer": is_skewer(board_before, move),
        "is_discovered_attack": is_discovered_attack(board_before, move),
        "is_double_attack": is_fork(board_before, move),  # same underlying concept
        "is_double_check": is_double_check(board_after),
        "is_mating_threat": is_mating_threat(board_after),
        "is_capture": captured is not None,
        "is_promotion": bool(move.promotion),
        "is_check": board_after.is_check(),
        "is_zwischenzug": is_zwischenzug(board_before, move),
        "is_x_ray": is_x_ray(board_before, move),
        "is_overloading": is_overloading(board_before, move),
        "is_sacrifice": is_sacrifice(board_before, move, eval_delta_cp),
        "is_deflection": is_deflection(board_before, move),
        "is_interposition": is_interposition(board_before, move),
    }
