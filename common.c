#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include "emmintrin.h"

#include "common.h"

void dump(char *arr, int n)
{
    int i;
    unsigned char *ptr = (unsigned char *)arr;

    for (i = 0; i < n; i++) {
        printf("%02x ", ptr[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }
    printf("\n");
}

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

#define _MALLOC_2D_BUFFER_SPACE (3*sizeof(int))
#define _MALLOC_3D_BUFFER_SPACE (4*sizeof(int))
void **malloc_2d(size_t x, size_t y, size_t type_size)
{
    int i;
    void **ret;
    int  **_2d;
    char *data_start;
    int *info;
    int alloc = ((x * y) * type_size) + y * sizeof(void *);

    if (x < 1 || y < 1)
        return NULL;

    ret = malloc(alloc + _MALLOC_2D_BUFFER_SPACE);
    _2d = (int **)ret;

    printf("mallocating %d bytes\n", alloc + _MALLOC_2D_BUFFER_SPACE);

    for (i = 0; i < y; i++) {
        data_start = (char *)((int **)_2d + y);
        data_start += _MALLOC_2D_BUFFER_SPACE;
        data_start += (i * x * type_size);

        printf("i: %d, ret: %p, data start: %p\n", i, ret, data_start);

        _2d[i] = (int *)data_start;
    }

    info = (int *)((char *)&_2d[0][0] - _MALLOC_2D_BUFFER_SPACE);

    info[0] = y;
    info[1] = x;
    info[2] = type_size;


    return ret;
}

int mem_2d_get_dims(void **mem, int *x, int *y, int *type_size)
{
    char **ret = (char **)mem;
    int *info;

    if (!mem)
        return 0;
    
    info = (int *)((char *)&ret[0][0] - _MALLOC_2D_BUFFER_SPACE);
    if (x)
        *x = info[1];
    if (y)
        *y = info[0];
    if (type_size)
        *type_size = info[2];

    return 1;
}

void **memdup_2d(void **mem)
{
    char *ptr;
    char **ret = (char **)mem;
    int *info = (int *)((char *)&ret[0][0] - _MALLOC_2D_BUFFER_SPACE);

    ptr = &ret[0][0];

    ret = (char **)malloc_2d(info[1], info[0], info[2]);
    if (ret == NULL)
        return NULL;

    // x * y * type_size * z
    memcpy(&ret[0][0], ptr, info[1] * info[0] * info[2]);

    return (void **)ret;
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

#ifdef DEBUG
    char ***tmp = (char ***)ret;
    char *tmpptr = (char *)ret;
    if (tmpptr + alloc + _MALLOC_3D_BUFFER_SPACE < &tmp[x - 1][y - 1][z - 1])
        printf("ERROR %s: something is wrong!\n", __FUNCTION__);
    int nx, ny, nz, nts;
    mem_3d_get_dims(ret, &nx, &ny, &nz, &nts);
    if (nx != x)
        printf("ERROR %s: something is wrong! nx != x\n", __FUNCTION__);
    if (ny != y)
        printf("ERROR %s: something is wrong! ny != y\n", __FUNCTION__);
    if (nz != z)
        printf("ERROR %s: something is wrong! nz != z\n", __FUNCTION__);
    if (nts != type_size)
        printf("ERROR %s: something is wrong! nts != type_size\n", __FUNCTION__);

    debug_print("allocating %d bytes @ %p\n", alloc + _MALLOC_3D_BUFFER_SPACE, ret);
#endif


    return ret;
}

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
#undef _MALLOC_2D_BUFFER_SPACE

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

float random_float_nr(void)
{
    float ret;
    static int urandom = -1;

    if (urandom == -1)
        urandom = open("/dev/urandom", O_RDONLY);

    read(urandom, &ret, sizeof(ret));

    return ret;
}

float random_float(void)
{
    return ((float)(unsigned)random_uint())/(float)0xffffffff;
}

// return a number r. min <= r <= max
int random_int_r(int min, int max)
{
    int ret;
    int nm = max - min;

    ret = random_uint() % (nm + 1);;

    if (min < 0 && max < 0)
        return -(ret - max) - 1;

    return ret + min;
}

int random_fill(void *arr, int n)
{
    static int urandom = -1;

    if (urandom == -1)
        urandom = open("/dev/urandom", O_RDONLY);

    return read(urandom, arr, n);
}


/* assumes arr is sorted */
int bisect(float *arr, float x, int n)
{
    int i = 0;

    while (i < n && x >= arr[i]) i++;

    if (i >= n)
        return n - 1;

    return i;
}

int *bitwise_and_sse2(int *a, int *b, int n, int *ret)
{
    char *aptr, *bptr, *retptr;
    unsigned int i;
    __m128i ad, bd, res;

    aptr = (char *)a;
    bptr = (char *)b;
    retptr = (char *)ret;

    for (i = 0; likely((i + ((128 / 8))) < n); i += (128 / 8)) {
        ad = _mm_loadu_si128((__m128i *)((char *)aptr + i));
        bd = _mm_loadu_si128((__m128i *)((char *)bptr + i));

        res = _mm_and_si128(ad, bd);
        _mm_storeu_si128((__m128i *)((char *)retptr + i), res);
    }

    for (; i < n; i++)
        retptr[i] = aptr[i] & bptr[i];

    return ret;
}

int *bitwise_or_sse2(int *a, int *b, int n, int *ret)
{
    char *aptr, *bptr, *retptr;
    unsigned int i;
    __m128i ad, bd, res;

    aptr = (char *)a;
    bptr = (char *)b;
    retptr = (char *)ret;

    for (i = 0; likely(i + (128 / 8) < n); i += (128 / 8)) {
        ad = _mm_loadu_si128((__m128i *)((char *)aptr + i));
        bd = _mm_loadu_si128((__m128i *)((char *)bptr + i));

        res = _mm_or_si128(ad, bd);
        _mm_storeu_si128((__m128i *)((char *)retptr + i), res);
    }

    for (; i < n; i++)
        retptr[i] = aptr[i] | bptr[i];

    debug_print("nice\n");

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

    if ((x[0] & x[1] & x[2]) != x[0]) {
        debug_print("a x = %d, b x = %d, res x = %d\n", x[0], x[1], x[2]);
        return 0; // x dimenstions are not of the same dimensions
    }

    if ((y[0] & y[1] & y[2]) != y[0]) {
        debug_print("a y = %d, b y = %d, res y = %d\n", y[0], y[1], y[2]);
        return 0; // y dimenstions are not of the same dimensions
    }

    if ((z[0] & z[1] & z[2]) != z[0]) {
        debug_print("a z = %d, b z = %d, res z = %d\n", z[0], z[1], z[2]);
        return 0; // y dimenstions are not of the same dimensions
    }

    if ((ts[0] & ts[1] & ts[2]) != ts[0]) {
        debug_print("a ts = %d, b ts = %d, res ts = %d\n", ts[0], ts[1], ts[2]);
        return 0; // types are not of same size
    }

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

    if ((x[0] & x[1] & x[2]) != x[0])
        return 0; // x dimenstions are not of the same dimensions

    if ((y[0] & y[1] & y[2]) != y[0])
        return 0; // y dimenstions are not of the same dimensions

    if ((z[0] & z[1] & z[2]) != z[0])
        return 0; // y dimenstions are not of the same dimensions

    if ((ts[0] & ts[1] & ts[2]) != ts[0])
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
    return get_moves_index(piece);
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
