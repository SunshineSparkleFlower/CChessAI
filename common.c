#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include "emmintrin.h"

#include "common.h"

static FILE *urandom = NULL;

void _dump(char *arr, int n)
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

    fprintf(stderr, "%s: ", function);
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

    for (i = 0; i < y; i++) {
        data_start = (char *)((int **)_2d + y);
        data_start += _MALLOC_2D_BUFFER_SPACE;
        data_start += (i * x * type_size);

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
/*
 * 
* 3D ARRAY START 
 * 
*/
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

    ret = (void ***)malloc(alloc + _MALLOC_3D_BUFFER_SPACE);
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

void memset_3d(void ***mem, int byte)
{
    int x, y, z, ts;
    char ***tmp;

    if (mem == NULL)
        return;

    mem_3d_get_dims(mem, &x, &y, &z, &ts);
    tmp = (char ***)mem;
    memset(&tmp[0][0][0], byte, (x * y * z) * ts);
}


#undef _MALLOC_3D_BUFFER_SPACE
#undef _MALLOC_2D_BUFFER_SPACE
/*
 * 
* 3D ARRAY END 
 * 
*/

int random_int(void)
{
    int ret;

    if (urandom == NULL)
        urandom = fopen("/dev/urandom", "r");

    if (fread(&ret, 1, sizeof(ret), urandom) != sizeof(ret))
        perror("fread");

    return ret;
}

unsigned random_uint(void)
{
    unsigned ret;

    if (urandom == NULL)
        urandom = fopen("/dev/urandom", "r");

    if (fread(&ret, 1, sizeof(ret), urandom) != sizeof(ret))
        perror("fread");

    return ret;
}

float random_float_nr(void)
{
    float ret;

    if (urandom == NULL)
        urandom = fopen("/dev/urandom", "r");

    if (fread(&ret, 1, sizeof(ret), urandom) != sizeof(ret))
        perror("fread");

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
    if (urandom == NULL)
        urandom = fopen("/dev/urandom", "r");

    return fread(arr, 1, n, urandom);
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

void shutdown(void)
{
    fclose(urandom);
}
