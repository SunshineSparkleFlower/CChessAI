import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cimport start
include "board.pxi"

cdef print_shit(board_t *boardc):
    print "number of legal moves: ", boardc.moves_count
    print "Legal moves:"
    for i in xrange(0, boardc.moves_count):
        print boardc.moves[i]
        #print "(%d, %d) -> (%d, %d)" % (boardc[i].moves.frm.y, boardc[i].moves.frm.x, boardc[i].moves.to.y, boardc[i].moves.to.x)

def test():

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] board = np.zeros((8,8), dtype=np.uint16)
    cdef coord_t coords
    cdef board_t cboard

    board.fill(P_EMPTY)

    coords.y = 0
    coords.x = 6

    board[1][0] = WHITE_PAWN
    board[0][5] = WHITE_KING
    board[0][6] = WHITE_ROOK
    board[0][7] = BLACK_ROOK

    board[6][0] = BLACK_PAWN
    board[7][0] = BLACK_ROOK
    board[7][7] = WHITE_PAWN
    board[4][4] = BLACK_KING
    

    cboard.board = <piece_t *>board.data
    cboard.moves_count = 0
    cboard.turn = WHITE

    #cdef move *moves = <move *>malloc(100 * sizeof(move))
    #cboard.moves = moves    
    #print "moves initialized"
    #print moves[1]
    
    print_board(cboard.board)
    get_legal_moves(&cboard, &coords)
    print "after get_legal_moves"
    #print_shit(&cboard)
    #print_legal_moves(&cboard)
    print_shit(&cboard)

test()
