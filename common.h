#ifndef __COMMON_H /* start of include guard */
#define __COMMON_H

#include <stdlib.h>
#include <stdint.h>

typedef uint16_t piece_t;

// y == down -> up, x == left -> right
typedef struct coord {
    int8_t y :4;
    int8_t x :4;
} coord_t;

typedef struct move {
    coord_t frm, to;
} move_t;

typedef struct legal_moves {
    move_t *moves;
    int num_moves;
} legal_moves_t;

typedef struct board {
    piece_t _board[8*8];
    piece_t *board_2d[8];
    piece_t *board; // only for backwards compatability. points to _board
    //piece_t *board;
    struct move moves[20*16];
    int moves_count;
    int turn;
}  board_t;

enum moves_index {
    PAWN = 0,
    ROOK,
    KNIGHT,
    BISHOP,
    QUEEN,
    KING,
    EMPTY = 12,
};

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define PREFETCH(addr) __builtin_prefetch(addr)
#define aligned(n) __attribute__((aligned(n)))
#define likely(x)	__builtin_expect(!!(x), 1)
#define unlikely(x)	__builtin_expect(!!(x), 0)


// returns the piece at board[row][col]
#define PIECE(board, row, col) *((board) + (((row) * 8) + col))
#define PIECE_ADDR (board, row, col) ((board) + (((row) * 8) + col))
#define for_each_board(board, ptr) \
    for ((ptr) = board; (ptr) <= ((board) + (((7) * 8) + 7)); ++(ptr))

#define BLACK -1
#define WHITE 1
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
#define P_EMPTY         (1 << 12)

#define enemy(board, row, col, turn) (color(PIECE(board, row, col)) * -1 == turn)
#define ally(board, row, col, turn) (color(PIECE(board, row, col)) == turn)
#define empty(board, row, col, turn) (color(PIECE(board, row, col)) == EMPTY)

#ifdef DEBUG
void _debug_print(const char *function, char *fmt, ...);
#define debug_print(fmt, ...) _debug_print(__FUNCTION__, fmt, ##__VA_ARGS__)
#else
#define debug_print(fmt, ...) 
#endif

extern struct board *create_board(char *fen);
extern void free_board(struct board *b);
extern int get_moves_index(piece_t piece);
extern int color(piece_t p);
extern enum moves_index get_piece_type(piece_t piece);

extern void ***malloc_3d(size_t x, size_t y, size_t z, size_t type_size);
extern void ***memdup_3d(void ***mem);
extern int mem_3d_get_dims(void ***mem, int *x, int *y, int *z, int *type_size);
extern int random_int(void);
extern unsigned random_uint(void);
extern int random_int_r(int min, int max);
extern float random_float(void);
extern int bisect(float *arr, float x, int n);
extern int *bitwise_and_sse2(int *a, int *b, int n, int *ret);
extern int bitwise_and_3d(void ***a, void ***b, void ***res);
extern int *bitwise_or_sse2(int *a, int *b, int n, int *ret);
extern int bitwise_or_3d(void ***a, void ***b, void ***res);
// works like numpy.random.choice 
// samples are the samples to fill ***out with
// n are the length of *samples
// ***out must be an array allocated with malloc_3d or memdup_3d
extern int choice_3d(uint16_t *samples, int n, uint16_t ***out);

extern coord_t move_offset[6][9][20];

#endif /* end of include guard: __COMMON_H */
