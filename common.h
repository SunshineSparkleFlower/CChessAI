#ifndef __COMMON_H /* start of include guard */
#define __COMMON_H

#include <stdint.h>

// y == down -> up, x == left -> right
typedef struct coord {
    int8_t y :4;
    int8_t x :4;
} coord_t;

typedef struct move {
    coord_t from, to;
} move_t;

typedef struct legal_moves {
    move_t *moves;
    int num_moves;
} legal_moves_t;

enum moves_index {
    EMPTY = 0,
    PAWN,
    ROOK,
    KNIGHT,
    BISHOP,
    QUEEN,
    KING,
};

typedef uint16_t piece_t;

#define BOARD(board, row, col) *((board) + (((row) * 8) + col))
#define for_each_board(board, ptr) \
    for ((ptr) = board; (ptr) <= ((board) + (((7) * 8) + 7)); ++(ptr))

#define BLACK -1
#define WHITE 1
//#define EMPTY           0
#define WHITE_PAWN      (1 << 0)
#define WHITE_ROOK      (1 << 1)
#define WHITE_KNIGHT    (1 << 2)
#define WHITE_BISHOP    (1 << 3)
#define WHITE_QUEEN     (1 << 4)
#define WHITE_KING      (1 << 5)

#define BLACK_PAWN      (1 << 6)
#define BLACK_ROOK      (1 << 7)
#define BLACK_KNIGHT    (1 << 8)
#define BLACK_BISHOP    (1 << 9)
#define BLACK_QUEEN     (1 << 10)
#define BLACK_KING      (1 << 11)

#define enemy(board, row, col) (color(BOARD(board, row, col)) * -1 == turn)
#define ally(board, row, col) (color(BOARD(board, row, col)) == turn)
#define empty(board, row, col) (color(BOARD(board, row, col)) == EMPTY)

extern int get_moves_index(piece_t piece);
extern enum moves_index get_piece_type(piece_t piece);
extern int color(piece_t p);

extern coord_t move_offset[6][9][20];

extern int turn;

extern piece_t board[8 * 8];
extern piece_t *board_2d[8];

#endif /* end of include guard: __COMMON_H */