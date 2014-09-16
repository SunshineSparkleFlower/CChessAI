cimport numpy as np

cdef extern from "rules.h":
    cdef struct coord:
        char y
        char x
    ctypedef coord coord_t
    ctypedef short piece_t
    cdef extern piece_t *board_2d[8]

    int get_legal_moves(piece_t *board[8], coord_t *from_c)


cdef np.ndarray board = np.zeros((8,8), dtype = np.uint16)

cdef coord_t coords
coords.y = 1
coords.x = 2
get_legal_moves(board_2d, &coords)
