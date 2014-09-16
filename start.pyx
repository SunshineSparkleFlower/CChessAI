import numpy as np
cimport numpy as np

cdef extern from "rules.h":
    cdef struct coord:
        char y
        char x
    ctypedef coord coord_t
    ctypedef short piece_t
    cdef extern piece_t *board_2d[8]

    int get_legal_moves(piece_t *, coord_t *from_c)
    extern void print_board(piece_t *board)


def detest():
    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] board = np.zeros((8,8), dtype = np.uint16)
    cdef coord_t coords
    coords.y = 1
    coords.x = 2

    get_legal_moves(<piece_t *>board.data, &coords)
    print_board(<piece_t *>board.data)

detest()
