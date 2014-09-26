import numpy as np
cimport numpy as np

cdef extern from "common.h":
    cdef struct coord:
        char y
        char x

    ctypedef coord coord_t
    ctypedef short piece_t

    cdef struct move:
        coord_t frm
        coord_t to

    cdef struct board:
        piece_t *board
        move moves[20*16]
        int moves_count
        int turn
    ctypedef board board_t

    cdef extern piece_t *board_2d[8]

    extern int get_legal_moves(board_t *, coord_t *)
    extern void print_board(piece_t *board)
    extern void print_legal_moves(board_t *board)

cdef print_shit(board_t *boardc):
    print boardc.moves_count
    for i in xrange(0, 2):
        print "(%d, %d) -> (%d, %d)" % (boardc[i].moves.frm.y, boardc[i].moves.frm.x, boardc[i].moves.to.y, boardc[i].moves.to.x)
    
def test():
    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] board = np.zeros((8,8), dtype=np.uint16)
    cdef coord_t coords
    cdef board_t cboard

    board.fill((1 << 12))

    coords.y = 1
    coords.x = 0

    board[1][0] = 1
    board[0][0] = 2

    board[6][0] = (1 << 6)
    board[7][0] = (1 << 7)

    cboard.board = <piece_t *>board.data
    cboard.moves_count = 0
    cboard.turn = 1

    print_board(cboard.board)
    get_legal_moves(&cboard, &coords)

    #print_shit(&cboard)
    print_legal_moves(&cboard)
    print "hahahahahahah"
    print_shit(&cboard)

test()
