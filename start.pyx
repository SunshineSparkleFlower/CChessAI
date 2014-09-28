import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

import cppmap
from cppmap import Memory
include "board.pxi"
include "AI.pxi"

cdef print_shit(board_t *boardc):
    print "number of legal moves: ", boardc.moves_count
    print "Legal moves:"
    for i in xrange(0, boardc.moves_count):
        print boardc.moves[i]



def get_best_move(Board board):
    pass
    

def test():
    board = Board("r6P/p7/7/4k3/8/8/P7/3qqKRr w - 0 1")
    board.print_board()
    board.calculate_legal_moves(0, 6)
    board.print_legal_moves()
    cdef int nr_shits = 5
    board.do_move(1,0,0,0)
    board.print_board()
    
    ai = AI()
    print ai.do_best_move(board)

#    memory.shits(nr_shits, m.data)  
#    board.calculate_legal_moves(0,7)
#    print board.get_legal_moves()    
#    board.print_legal_moves()
#    print "finding all legal moves"
#    legalmoves = board.get_all_legal_moves()
#    print legalmoves
#    board.print_board()
#    board.move(legalmoves[0])
#    board.print_board()
#    board.reverse_move()
#    board.print_board()
    
    #print board.multiply(tes[2])

#    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] board = np.zeros((8,8), dtype=np.uint16)
#    cdef coord_t coords
#    cdef board_t cboard
#
#    board.fill(P_EMPTY)
#
#    coords.y = 0
#    coords.x = 6
#
#    board[1][0] = WHITE_PAWN
#    board[0][5] = WHITE_KING
#    board[0][6] = WHITE_ROOK
#    board[0][7] = BLACK_ROOK
#
#    board[6][0] = BLACK_PAWN
#    board[7][0] = BLACK_ROOK
#    board[7][7] = WHITE_PAWN
#    board[4][4] = BLACK_KING
#    
#
#    cboard.board = <piece_t *>board.data
#    cboard.moves_count = 0
#    cboard.turn = WHITE
#
#    #cdef move *moves = <move *>malloc(100 * sizeof(move))
#    #cboard.moves = moves    
#    #print "moves initialized"
#    #print moves[1]
#    
#    print_board(cboard.board)
#    get_legal_moves(&cboard, &coords)
#    print "after get_legal_moves"
#    #print_shit(&cboard)
#    print_shit(&cboard)
