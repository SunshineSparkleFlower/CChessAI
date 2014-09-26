#include <stdlib.h>
#include "common.h"

struct board *create_board(char *fen)
{
    struct board *ret = calloc(1, sizeof(struct board));

    return ret;
}

void free_board(struct board *b)
{
    free(b);
}

int get_moves_index(piece_t piece)
{
    unsigned int v = (((piece >> 6) | piece) & 0x3f) | (piece & P_EMPTY) ; // find the number of trailing zeros in 32-bit v 
    int r;           // result goes here
    static const int MultiplyDeBruijnBitPosition[32] = 
    {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };
    r = MultiplyDeBruijnBitPosition[((uint32_t)((v & -v) * 0x077CB531U)) >> 27];

    return r;
}

int color(uint16_t p)
{
    if (p & P_EMPTY)
        return EMPTY;

    register int ret = !(p & ((1 << 6) - 1));

    return !ret - ret;
}

enum moves_index get_piece_type(piece_t piece)
{
    register unsigned ret = get_moves_index(piece);
    return ret;
}

coord_t move_offset[6][9][20] = {
    // pawn
    {{{1, 0}, {1, 1}, {1, -1}, {0, 0}}, {{0, 0}}},

    // rook
    {{{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {0, 0}},
    {{-1, 0}, {-2, 0}, {-3, 0}, {-4, 0}, {-5, 0}, {-6, 0}, {-7, 0}, {0, 0}},
    {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 0}},
    {{0, -1}, {0, -2}, {0, -3}, {0, -4}, {0, -5}, {0, -6}, {0, -7}, {0, 0}},
    {{0, 0}}},

    // knight
    {{{2, -1}, {2, 1}, {1, 2}, {-1, 2}, {0, 0}},
    {{-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {0, 0}},
    {{0, 0}}},

    // bishop
    {{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {0, 0}},
    {{1, -1}, {2, -2}, {3, -3}, {4, -4}, {5, -5}, {6, -6}, {7, -7}, {0, 0}},
    {{-1, 1}, {-2, 2}, {-3, 3}, {-4, 4}, {-5, 5}, {-6, 6}, {-7, 7}, {0, 0}},
    {{-1, -1}, {-2, -2}, {-3, -3}, {-4, -4}, {-5, -5}, {-6, -6}, {-7, -7}, {0, 0}},
    {{0, 0}}},
    
    // queen
    {{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {0, 0}},
    {{1, -1}, {2, -2}, {3, -3}, {4, -4}, {5, -5}, {6, -6}, {7, -7}, {0, 0}},
    {{-1, 1}, {-2, 2}, {-3, 3}, {-4, 4}, {-5, 5}, {-6, 6}, {-7, 7}, {0, 0}},
    {{-1, -1}, {-2, -2}, {-3, -3}, {-4, -4}, {-5, -5}, {-6, -6}, {-7, -7}, {0, 0}},
    {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {0, 0}},
    {{-1, 0}, {-2, 0}, {-3, 0}, {-4, 0}, {-5, 0}, {-6, 0}, {-7, 0}, {0, 0}},
    {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 0}},
    {{0, -1}, {0, -2}, {0, -3}, {0, -4}, {0, -5}, {0, -6}, {0, -7}, {0, 0}},
    {{0, 0}}},

    // king
    {{{1, 0}, {0, 0}}, {{-1, 0}, {0, 0}}, {{0, 1}, {0, 0}}, {{0, -1}, {0, 0}},
    {{1, -1}, {0, 0}}, {{1, 1}, {0, 0}}, {{-1, -1}, {0, 0}}, {{-1, 1}, {0, 0}},
    {{0, 0}}},
};

int turn = WHITE;

/*
piece_t board[8 * 8] = {
    WHITE_ROOK, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING, WHITE_BISHOP, WHITE_KNIGHT, WHITE_ROOK,
    WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN, WHITE_PAWN,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN, BLACK_PAWN,
    BLACK_ROOK, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING, BLACK_BISHOP, BLACK_KNIGHT, BLACK_ROOK,
};
*/

piece_t board[8 * 8] = {
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, BLACK_ROOK, WHITE_QUEEN, WHITE_KING, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, WHITE_PAWN, BLACK_PAWN, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, BLACK_KING, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
    P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY, P_EMPTY,
};

piece_t *board_2d[8] = {&board[0], &board[8 * 1], &board[8 * 2], &board[8 * 3],
    &board[8 * 4], &board[8 * 5], &board[8 * 6], &board[8 * 7]};
