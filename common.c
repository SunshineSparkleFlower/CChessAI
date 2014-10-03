#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include "emmintrin.h"

#include "common.h"


void _debug_print(const char *function, char *fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);

    fprintf(stderr, "[DEBUG] %s: ", function);
    vfprintf(stderr, fmt, ap);

    va_end(ap);
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

#define _MALLOC_3D_BUFFER_SPACE (4*sizeof(int))
int mem_3d_get_dims(void ***mem, int *x, int *y, int *z, int *type_size)
{
    char ***ret = (char ***)mem;
    int *info;

    if (!mem)
        return 0;

    info = (int *)((char *)&ret[0][0][0] - _MALLOC_3D_BUFFER_SPACE);
    if (x)
        *x = info[2];
    if (y)
        *y = info[1];
    if (z)
        *z = info[0];
    if (type_size)
        *type_size = info[3];

    return 1;
}

void ***malloc_3d(size_t x, size_t y, size_t z, size_t type_size)
{
    int i, j;
    void ***ret;
    int ***_3d, **_2d;
    char *data_start;
    int *info;
    int alloc = ((x * y * z) * type_size) +  ((y * z) + z) * sizeof(void *);

    if (x < 1 || y < 1 || z < 1)
        return NULL;

    ret = malloc(alloc + _MALLOC_3D_BUFFER_SPACE);
    _3d = (int ***)ret;

    for (i = 0; i < z; i++) {
        _2d = (int **)_3d + z + (y * i);
        data_start = (char *)((int **)_3d + z + (y * z));
        data_start += _MALLOC_3D_BUFFER_SPACE;
        data_start += (i * x * y * type_size);

        for (j = 0; j < y; j++)
            _2d[j] = (int *)((char *)data_start + (j * x * type_size));

        _3d[i] = _2d;
    }

    info = (int *)((char *)&_3d[0][0][0] - _MALLOC_3D_BUFFER_SPACE);

    info[0] = z;
    info[1] = y;
    info[2] = x;
    info[3] = type_size;

    return ret;
}

void ***memdup_3d(void ***mem)
{
    char *ptr;
    char ***ret = (char ***)mem;
    int *info = (int *)((char *)&ret[0][0][0] - _MALLOC_3D_BUFFER_SPACE);

    ptr = &ret[0][0][0];

    ret = (char ***)malloc_3d(info[2], info[1], info[0], info[3]);
    if (ret == NULL)
        return NULL;

    // x * y * type_size * z
    memcpy(&ret[0][0][0], ptr, (info[2] * info[1] * info[3]) * info[0]);

    return (void ***)ret;
}
#undef _MALLOC_3D_BUFFER_SPACE

int random_int(void)
{
    int ret;
    static int urandom = -1;

    if (urandom == -1)
        urandom = open("/dev/urandom", O_RDONLY);

    read(urandom, &ret, sizeof(ret));

    return ret;
}

unsigned random_uint(void)
{
    unsigned ret;
    static int urandom = -1;

    if (urandom == -1)
        urandom = open("/dev/urandom", O_RDONLY);

    read(urandom, &ret, sizeof(ret));

    return ret;
}

float random_float(void)
{
    float ret;
    static int urandom = -1;

    if (urandom == -1)
        urandom = open("/dev/urandom", O_RDONLY);

    read(urandom, &ret, sizeof(ret));

    return ret;
}

// return a number r. min <= r <= max
int random_int_r(int min, int max)
{
    int ret;
    int nm = max - min;

    ret = random_int() % (nm + 1);;

    if (min < 0 && max < 0)
        return -(ret - max) - 1;

    return ret + min;
}

/* assumes arr is sorted */
int bisect(float *arr, float x, int n)
{
    int i = 0;

    while (i < n && x <= arr[i++]);

    if (i >= n)
        i = n - 1;

    return i;
}

int *bitwise_and_sse2(int *a, int *b, int n, int *ret)
{
    unsigned int i;
    __m128i ad, bd, res;

    for (i = 0; likely((i + ((128 / 8) / 4)) < n); i += (128 / 8) / 4) {
        ad = _mm_loadu_si128((__m128i *)(a + i));
        bd = _mm_loadu_si128((__m128i *)(b + i));

        res = _mm_and_si128(ad, bd);
        _mm_storeu_si128((__m128i *)(ret + i), res);
    }

    for (; likely(i < n); ++i)
        ret[i] = a[i] & b[i];

    return ret;
}

int *bitwise_or_sse2(int *a, int *b, int n, int *ret)
{
    unsigned int i;
    __m128i ad, bd, res;

    for (i = 0; likely((i + ((128 / 8) / 4)) < n); i += (128 / 8) / 4) {
        ad = _mm_loadu_si128((__m128i *)(a + i));
        bd = _mm_loadu_si128((__m128i *)(b + i));

        res = _mm_or_si128(ad, bd);
        _mm_storeu_si128((__m128i *)(ret + i), res);
    }

    for (; likely(i < n); ++i)
        ret[i] = a[i] | b[i];

    return ret;
}

/* wrapper function for bitwise_or_sse2. a, b, and res must be of the same
 * dimensions and types.
 * ***a, ***b and ***res must be arrays allocated by malloc_3d, or memdup_3d
 * returns 1 on success, 0 otherwise
 */
int bitwise_or_3d(void ***a, void ***b, void ***res)
{
    char ***ptra, ***ptrb, ***ptrres;
    int x[3], y[3], z[3], ts[3];

    ptra = (char ***)a;
    ptrb = (char ***)b;
    ptrres = (char ***)res;

    mem_3d_get_dims(a, &x[0], &y[0], &z[0], &ts[0]);
    mem_3d_get_dims(b, &x[1], &y[1], &z[1], &ts[1]);
    mem_3d_get_dims(res, &x[2], &y[2], &z[2], &ts[2]);

    if ((x[0] ^ x[1] ^ x[2]))
        return 0; // x dimenstions are not of the same dimensions

    if ((y[0] ^ y[1] ^ y[2]))
        return 0; // y dimenstions are not of the same dimensions

    if ((z[0] ^ z[1] ^ z[2]))
        return 0; // y dimenstions are not of the same dimensions

    if ((ts[0] ^ ts[1] ^ ts[2]))
        return 0; // types are not of same size

    bitwise_or_sse2((int *)&ptra[0][0][0], (int *)&ptrb[0][0][0],
            x[0] * y[0] * ts[0] * z[0], (int *)&ptrres[0][0][0]);

    return 1;
}

/* wrapper function for bitwise_and_sse2. a, b, and res must be of the same
 * dimensions and types.
 * ***a, ***b and ***res must be arrays allocated by malloc_3d, or memdup_3d
 * returns 1 on success, 0 otherwise
 */
int bitwise_and_3d(void ***a, void ***b, void ***res)
{
    char ***ptra, ***ptrb, ***ptrres;
    int x[3], y[3], z[3], ts[3];

    ptra = (char ***)a;
    ptrb = (char ***)b;
    ptrres = (char ***)res;

    mem_3d_get_dims(a, &x[0], &y[0], &z[0], &ts[0]);
    mem_3d_get_dims(b, &x[1], &y[1], &z[1], &ts[1]);
    mem_3d_get_dims(res, &x[2], &y[2], &z[2], &ts[2]);

    if ((x[0] ^ x[1] ^ x[2]))
        return 0; // x dimenstions are not of the same dimensions

    if ((y[0] ^ y[1] ^ y[2]))
        return 0; // y dimenstions are not of the same dimensions

    if ((z[0] ^ z[1] ^ z[2]))
        return 0; // y dimenstions are not of the same dimensions

    if ((ts[0] ^ ts[1] ^ ts[2]))
        return 0; // types are not of same size

    bitwise_and_sse2((int *)&ptra[0][0][0], (int *)&ptrb[0][0][0],
            x[0] * y[0] * ts[0] * z[0], (int *)&ptrres[0][0][0]);

    return 1;
}

// works like numpy.random.choice 
// samples are the samples to fill ***out with
// n are the length of *samples
// ***out must be an array allocated with malloc_3d or memdup_3d
int choice_3d(uint16_t *samples, int n, uint16_t ***out)
{
    int i, j, k;
    int x, y, z;

    if (!mem_3d_get_dims((void ***)out, &x, &y, &z, NULL))
        return 0;

    for (i = 0; i < z; i++)
        for (j = 0; j < y; j++)
            for (k = 0; k < x; k++)
                out[i][j][k] = samples[random_int_r(0, n - 1)];

    return 1;
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
