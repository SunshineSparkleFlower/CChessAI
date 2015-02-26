#ifndef __COMMON_H /* start of include guard */
#define __COMMON_H

#include <stdlib.h>
#include <stdint.h>

typedef uint64_t u64;

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

#define __inline __attribute__((always_inline))
#define is_set(b, n) ((b) & (1lu << (n)))
#define set_bit(b, n) ((b) |= (1lu << (n)))
#define set_mask(b, m) ((b) |= (m))
#define isolate_bit(b, n) ((b) & (1lu << (n)))
#define clear_bit(b, n) ((b) &= ~(1lu << (n)))


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

#if defined(DEBUG)
void _debug_print(const char *function, char *fmt, ...);
#define debug_print(fmt, ...) _debug_print(__FUNCTION__, "[DEBUG] " fmt, ##__VA_ARGS__)
#else
#define debug_print(fmt, ...)
#endif

#include "board.h"
extern int get_moves_index(piece_t piece);
extern int color(piece_t p);
extern enum moves_index get_piece_type(piece_t piece);

extern void **malloc_2d(size_t x, size_t y, size_t type_size);
extern int mem_2d_get_dims(void **mem, int *x, int *y, int *type_size);
extern void **memdup_2d(void **mem);
extern void ***malloc_3d(size_t x, size_t y, size_t z, size_t type_size);
extern void ***memdup_3d(void ***mem);
extern void memset_3d(void ***mem, int byte);
extern int mem_3d_get_dims(void ***mem, int *x, int *y, int *z, int *type_size);
extern int max(int a, int b);
extern int min(int a, int b);
extern int keep_in_range(int val, int min_val, int max_val);
extern int random_int(void);
extern unsigned random_uint(void);
extern int random_int_r(int min, int max);
extern unsigned random_uint(void);
extern float random_float(void);
extern int random_fill(void *arr, int n);
extern int bisect(float *arr, float x, int n);
extern int *bitwise_and_sse2(int *a, int *b, int n, int *ret);
extern int bitwise_and_3d(void ***a, void ***b, void ***res);
extern int *bitwise_or_sse2(int *a, int *b, int n, int *ret);
extern int bitwise_or_3d(void ***a, void ***b, void ***res);
#define dump(arr, n) _dump((char *)(arr), (n))
extern void _dump(char *arr, int n);
extern int choice_3d(uint16_t *samples, int n, uint16_t ***out);
extern void _shutdown(void);

#endif /* end of include guard: __COMMON_H */
