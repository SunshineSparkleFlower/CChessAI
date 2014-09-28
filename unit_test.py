#!/usr/bin/python

import start
import sys

def error(board, msg):
    board.print_board()
    print msg
    print "legal_moves are:"
    print_legal_moves(board.get_all_legal_moves())
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

def print_legal_moves(legal_moves):
    for l in legal_moves:
        frm = to_tuple(l['frm'])
        to = to_tuple(l['to'])
        print "%s -> %s" % (str(frm), str(to))

# test 1
fen = "8/R5p1/4pp2/k1BpP3/P2P1P2/3r1PK1/7P/8 b - - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()
if len(legal_moves) != 0:
    error(board, "number of legal moves should be 0")


# test 2
fen = "8/R5p1/2q1pp2/k1BpP3/P2P1P2/3r1PK1/7P/8 b - - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()

if len(legal_moves) != 1:
    error(board, "number of legal moves should be 1")

if not move_in_list((5, 2), (5, 0), legal_moves):
    error(board, "move from " + str((5, 2)) + " to " + str((5, 0)) + " should be possible")


# test 3
fen = "8/8/4k3/8/4K3/8/8/8 w - - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()
if len(legal_moves) != 5:
    error(board, "number of legal moves should be 5")

legals = [[(3, 4), (3, 3)], [(3, 4), (2, 3)], [(3, 4), (2, 4)], [(3, 4), (2, 5)], [(3, 4), (3, 5)]]
for l in legals:
    if not move_in_list(l[0], l[1], legal_moves):
        error(board, "move from %s to %s should be possible" % (str(l[0]), str(l[1])))


# test 4
fen = "8/8/4k3/8/2ppK3/8/8/8 w - - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()
if len(legal_moves) != 3:
    error(board, "number of legal moves should be 3 - is %d" % len(legal_moves))

legals = [[(3, 4), (3, 3)], [(3, 4), (2, 5)], [(3, 4), (3, 5)]]
for l in legals:
    if not move_in_list(l[0], l[1], legal_moves):
        error(board, "move from %s to %s should be possible" % (str(l[0]), str(l[1])))


# test 5
fen = "8/8/4k3/2p5/2ppK3/8/8/8 w - - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()
if len(legal_moves) != 2:
    error(board, "number of legal moves should be 2 - is %d" % len(legal_moves))

legals = [[(3, 4), (2, 5)], [(3, 4), (3, 5)]]
for l in legals:
    if not move_in_list(l[0], l[1], legal_moves):
        error(board, "move from %s to %s should be possible" % (str(l[0]), str(l[1])))


# test 6 (starting position)
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = start.Board(fen)
legal_moves = board.get_all_legal_moves()

legals =  [[(1, 0), (2, 0)], [(1, 1), (2, 1)], [(1, 2), (2, 2)], [(1, 3), (2, 3)], [(1, 4), (2, 4)], [(1, 5), (2, 5)], [(1, 6), (2, 6)], [(1, 7), (2, 7)]]
legals += [[(1, 0), (3, 0)], [(1, 1), (3, 1)], [(1, 2), (3, 2)], [(1, 3), (3, 3)], [(1, 4), (3, 4)], [(1, 5), (3, 5)], [(1, 6), (3, 6)], [(1, 7), (3, 7)]]
legals += [[(0, 1), (2, 0)], [(0, 1), (2, 2)], [(0, 6), (2, 5)], [(0, 6), (2, 7)]]
if len(legal_moves) != len(legals):
    error(board, "number of legal moves should be %d - is %d" % (len(legal_moves)), len(legals))

for l in legals:
    if not move_in_list(l[0], l[1], legal_moves):
        error(board, "move from %s to %s should be possible" % (str(l[0]), str(l[1])))

print "All tests passed!"
