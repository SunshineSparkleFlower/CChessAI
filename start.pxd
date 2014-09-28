cdef extern from "common.h":
    cdef struct coord:
        char y
        char x

    ctypedef coord coord_t
    ctypedef short piece_t

    cdef struct move:
        coord_t frm
        coord_t to
    ctypedef move move_t
    
    cdef struct board:
        piece_t *board
        move moves[20*16];    
        int moves_count
        int turn
    ctypedef board board_t

    
    cdef short BLACK = -1
    cdef short WHITE = 1
    cdef short WHITE_PAWN = (1 << 0)
    cdef short WHITE_ROOK = (1 << 1)
    cdef short WHITE_KNIGHT = (1 << 2)
    cdef short WHITE_BISHOP = (1 << 3)
    cdef short WHITE_QUEEN = (1 << 4)
    cdef short WHITE_KING = (1 << 5)

    cdef short BLACK_PAWN = (1 << 6)
    cdef short BLACK_ROOK = (1 << 7)
    cdef short BLACK_KNIGHT = (1 << 8)
    cdef short BLACK_BISHOP = (1 << 9)
    cdef short BLACK_QUEEN = (1 << 10)
    cdef short BLACK_KING = (1 << 11)
    cdef short P_EMPTY = (1 << 12)

cdef extern from "rules.h":
    extern int get_legal_moves(board_t *, coord_t *)
    extern void print_board(piece_t *board)
    extern void print_legal_moves(board_t *board)
    extern void do_move(piece_t *board, coord_t frm, coord_t to, piece_t *backup)
    extern board_t *get_all_legal_moves(board_t *board_struct)
    extern void reverse_move(piece_t *board, coord_t frm, coord_t to, piece_t backup)
