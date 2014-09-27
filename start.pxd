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
        move moves[20*16];    
        int moves_count
        int turn
    ctypedef board board_t

    cdef int BLACK = -1
    cdef int WHITE = 1
    cdef int WHITE_PAWN = (1 << 0)
    cdef int WHITE_ROOK = (1 << 1)
    cdef int WHITE_KNIGHT = (1 << 2)
    cdef int WHITE_BISHOP = (1 << 3)
    cdef int WHITE_QUEEN = (1 << 4)
    cdef int WHITE_KING = (1 << 5)

    cdef int BLACK_PAWN = (1 << 6)
    cdef int BLACK_ROOK = (1 << 7)
    cdef int BLACK_KNIGHT = (1 << 8)
    cdef int BLACK_BISHOP = (1 << 9)
    cdef int BLACK_QUEEN = (1 << 10)
    cdef int BLACK_KING = (1 << 11)
    cdef int P_EMPTY = (1 << 12)

cdef extern from "rules.h":
    extern int get_legal_moves(board_t *, coord_t *)
    extern void print_board(piece_t *board)
    extern void print_legal_moves(board_t *board)

