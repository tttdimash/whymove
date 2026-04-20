"""FeatureExtractor — orchestrates board_utils, tactical, and engine calls."""

from __future__ import annotations

import chess
import chess.pgn
import io

from whymove.engine.base import ChessEngine
from whymove.features.board_utils import (
    chebyshev_distance,
    count_center_squares_controlled,
    count_doubled_pawns,
    count_isolated_pawns,
    count_key_squares_controlled,
    count_passed_pawns,
    count_pawn_islands,
    fen_after_move,
    get_king_zone_attackers,
    get_material_value,
    get_open_files_near_king,
    get_piece_mobility,
    get_total_mobility,
    normalize_eval_to_white,
)
from whymove.features.tactical import compute_tactical_flags
from whymove.models import (
    KingSafetyFeatures,
    PawnStructureFeatures,
    PieceInfo,
    PositionFeatures,
    TacticalFlags,
)


class FeatureExtractor:
    """Extract a PositionFeatures object from a FEN + move UCI."""

    def __init__(self, engine: ChessEngine) -> None:
        self._engine = engine

    def extract(
        self,
        fen: str,
        move_uci: str,
        engine_depth: int = 20,
    ) -> PositionFeatures:
        board_before = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        fen_after = fen_after_move(fen, move_uci)
        board_after = chess.Board(fen_after)
        mover_color = board_before.turn

        # ── Evaluation ──────────────────────────────────────────────────────
        eval_before = self._engine.evaluate(fen, engine_depth)
        eval_after = self._engine.evaluate(fen_after, engine_depth)

        eval_before_cp = normalize_eval_to_white(eval_before.score_cp, board_before.turn)
        eval_after_cp = normalize_eval_to_white(eval_after.score_cp, board_after.turn)

        # Handle mate scores: use large sentinel value
        if eval_before.score_mate is not None:
            eval_before_cp = 10000 * (1 if eval_before.score_mate > 0 else -1)
            if board_before.turn == chess.BLACK:
                eval_before_cp = -eval_before_cp
        if eval_after.score_mate is not None:
            eval_after_cp = 10000 * (1 if eval_after.score_mate > 0 else -1)
            if board_after.turn == chess.BLACK:
                eval_after_cp = -eval_after_cp

        eval_delta_cp = eval_after_cp - eval_before_cp

        # ── Material ─────────────────────────────────────────────────────────
        mat_before = get_material_value(board_before)
        mat_after = get_material_value(board_after)
        mat_delta = mat_after - mat_before

        # ── Mobility ─────────────────────────────────────────────────────────
        mob_before = get_piece_mobility(board_before, move.from_square)
        mob_after = get_piece_mobility(board_after, move.to_square)
        white_mob_before = get_total_mobility(board_before, chess.WHITE)
        white_mob_after = get_total_mobility(board_after, chess.WHITE)
        black_mob_before = get_total_mobility(board_before, chess.BLACK)
        black_mob_after = get_total_mobility(board_after, chess.BLACK)

        # ── Piece info ───────────────────────────────────────────────────────
        piece = board_before.piece_at(move.from_square)
        assert piece is not None, f"No piece at {move.from_square} in {fen}"
        piece_info = PieceInfo(
            piece_type=piece.piece_type,
            color=piece.color,
            from_square=move.from_square,
            to_square=move.to_square,
        )
        distance = chebyshev_distance(move.from_square, move.to_square)

        # ── King safety ──────────────────────────────────────────────────────
        white_zone_before = get_king_zone_attackers(board_before, chess.WHITE)
        white_zone_after = get_king_zone_attackers(board_after, chess.WHITE)
        black_zone_before = get_king_zone_attackers(board_before, chess.BLACK)
        black_zone_after = get_king_zone_attackers(board_after, chess.BLACK)
        white_open_before = get_open_files_near_king(board_before, chess.WHITE)
        white_open_after = get_open_files_near_king(board_after, chess.WHITE)
        black_open_before = get_open_files_near_king(board_before, chess.BLACK)
        black_open_after = get_open_files_near_king(board_after, chess.BLACK)

        king_safety = KingSafetyFeatures(
            white_king_zone_attacks_delta=white_zone_after - white_zone_before,
            black_king_zone_attacks_delta=black_zone_after - black_zone_before,
            white_open_files_near_king_delta=white_open_after - white_open_before,
            black_open_files_near_king_delta=black_open_after - black_open_before,
        )

        # ── Pawn structure ───────────────────────────────────────────────────
        # From mover's perspective
        c = mover_color
        pawn_structure = PawnStructureFeatures(
            doubled_pawns_delta=count_doubled_pawns(board_after, c) - count_doubled_pawns(board_before, c),
            isolated_pawns_delta=count_isolated_pawns(board_after, c) - count_isolated_pawns(board_before, c),
            passed_pawns_delta=count_passed_pawns(board_after, c) - count_passed_pawns(board_before, c),
            pawn_islands_delta=count_pawn_islands(board_after, c) - count_pawn_islands(board_before, c),
        )

        # ── Square control ───────────────────────────────────────────────────
        center_delta = (
            count_center_squares_controlled(board_after, mover_color)
            - count_center_squares_controlled(board_before, mover_color)
        )
        key_sq_delta = (
            count_key_squares_controlled(board_after, mover_color)
            - count_key_squares_controlled(board_before, mover_color)
        )

        # ── Tactical flags ───────────────────────────────────────────────────
        tactical_dict = compute_tactical_flags(board_before, board_after, move, eval_delta_cp)
        tactical = TacticalFlags(**tactical_dict)

        return PositionFeatures(
            eval_before_cp=eval_before_cp,
            eval_after_cp=eval_after_cp,
            eval_delta_cp=eval_delta_cp,
            material_before=mat_before,
            material_after=mat_after,
            material_delta=mat_delta,
            moved_piece_mobility_before=mob_before,
            moved_piece_mobility_after=mob_after,
            total_white_mobility_delta=white_mob_after - white_mob_before,
            total_black_mobility_delta=black_mob_after - black_mob_before,
            piece=piece_info,
            distance_moved=distance,
            destination_rank=chess.square_rank(move.to_square),
            destination_file=chess.square_file(move.to_square),
            king_safety=king_safety,
            pawn_structure=pawn_structure,
            center_control_delta=center_delta,
            key_square_control_delta=key_sq_delta,
            tactical=tactical,
        )
