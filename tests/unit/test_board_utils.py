"""Unit tests for board_utils.py."""

from __future__ import annotations

import chess
import pytest

from whymove.features.board_utils import (
    chebyshev_distance,
    count_doubled_pawns,
    count_isolated_pawns,
    count_passed_pawns,
    count_pawn_islands,
    fen_after_move,
    get_material_value,
    get_piece_mobility,
    normalize_eval_to_white,
)


def test_material_value_starting_position():
    board = chess.Board()
    assert get_material_value(board) == 0  # Equal material at start


def test_material_value_after_capture():
    # White captures a pawn
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    board_after = board.copy()
    board_after.push_uci("e4d5")
    # White gained a pawn (100cp)
    assert get_material_value(board_after) == 100


def test_chebyshev_distance_adjacent():
    assert chebyshev_distance(chess.E4, chess.E5) == 1
    assert chebyshev_distance(chess.E4, chess.D5) == 1


def test_chebyshev_distance_knight_move():
    assert chebyshev_distance(chess.G1, chess.F3) == 2


def test_chebyshev_distance_same_square():
    assert chebyshev_distance(chess.E4, chess.E4) == 0


def test_piece_mobility_knight_center():
    board = chess.Board("8/8/8/8/4N3/8/8/4K3 w - - 0 1")
    # Knight on e4 has 8 possible moves (all legal in this empty board)
    mob = get_piece_mobility(board, chess.E4)
    assert mob == 8


def test_piece_mobility_no_piece():
    board = chess.Board()
    assert get_piece_mobility(board, chess.E4) == 0


def test_doubled_pawns():
    # Two white pawns on the e file
    board = chess.Board("8/8/8/8/4P3/4P3/8/4K3 w - - 0 1")
    assert count_doubled_pawns(board, chess.WHITE) == 1
    assert count_doubled_pawns(board, chess.BLACK) == 0


def test_no_doubled_pawns_starting():
    board = chess.Board()
    assert count_doubled_pawns(board, chess.WHITE) == 0
    assert count_doubled_pawns(board, chess.BLACK) == 0


def test_isolated_pawns():
    # White pawn on a4 with no pawns on b or h files
    board = chess.Board("8/8/8/8/P7/8/8/4K3 w - - 0 1")
    assert count_isolated_pawns(board, chess.WHITE) == 1


def test_passed_pawns():
    # White pawn on e5 with no black pawns on d, e, f files ahead
    board = chess.Board("8/8/8/4P3/8/8/8/4K1k1 w - - 0 1")
    assert count_passed_pawns(board, chess.WHITE) == 1


def test_pawn_islands_single():
    board = chess.Board("8/8/8/8/PPP5/8/8/4K3 w - - 0 1")
    assert count_pawn_islands(board, chess.WHITE) == 1


def test_pawn_islands_two():
    # Pawns on a, b (island 1) and e, f (island 2)
    board = chess.Board("8/8/8/8/PP2PP2/8/8/4K3 w - - 0 1")
    assert count_pawn_islands(board, chess.WHITE) == 2


def test_fen_after_move():
    board = chess.Board()
    fen_after = fen_after_move(chess.STARTING_FEN, "e2e4")
    board.push_uci("e2e4")
    assert fen_after == board.fen()


def test_normalize_eval_white_turn():
    assert normalize_eval_to_white(50, chess.WHITE) == 50


def test_normalize_eval_black_turn():
    # Black's turn: +50 from black's perspective = -50 white-relative
    assert normalize_eval_to_white(50, chess.BLACK) == -50
