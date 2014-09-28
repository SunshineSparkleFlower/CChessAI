#!/usr/bin/python

import start
import sys

def error(board, msg):
    board.print_board()
    print msg
    sys.exit(1)

def to_tuple(move):
    return move['y'], move['x']
    
def move_in_list(frm, to, lst):
    for m in lst:
        lfrm = to_tuple(m['frm'])
        lto = to_tuple(m['to'])

        if lfrm == frm and lto == to:
            return True

    return False

fen = "8/R5p1/4pp2/k1BpP3/P2P1P2/3r1PK1/7P/8 b - - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()
if len(legal_moves) != 0:
    error(board, "number of legal moves should be 0")


fen = "8/R5p1/2q1pp2/k1BpP3/P2P1P2/3r1PK1/7P/8 b - - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()

if len(legal_moves) != 1:
    error(board, "number of legal moves should be 1")

if not move_in_list((5, 2), (5, 0), legal_moves):
    error(board, "move from " + str((5, 2)) + " to " + str((5, 0)) + " should be possible")
