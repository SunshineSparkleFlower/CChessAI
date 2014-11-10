#include <stdio.h>
#include <unistd.h>

#include "common.h"
#include "AI.h"
#include "board.h"

__device__ int result;


/*
__device__ void cunand(unsigned char *V, unsigned char *brain, int size, unsigned char *board, int board_size)
{
    int index = threadIdx.x, i;
    __shared__ int found;

    found = 0;

    __syncthreads();
    if (isolate_bit(V[index / 8], index % 8) &
            isolate_bit(brain[index / 8], index % 8)) {
        set_bit(V[index / 8], blockIdx.x);
        found = 1;
        return;
    }

    for (i = index * 2 * 8; !found && i < (index * 2 * 8) + (board_size)/gridDim.x; ++i)
        if (isolate_bit(board[index / 8], i % 8) &
                isolate_bit(brain[(index + size) / 8], (index + size) % 8)) {
            set_bit(V[index / 8], blockIdx.x);
            found = 1;
            return;
        }
}

// one thread block pr. port. board_size/16 threads pr thread block
__global__ void cuscore(unsigned char *brain, unsigned char *board, int
        board_size, int nr_synapsis, unsigned char *V)
{
    //__shared__ unsigned char V[128/8];

    cunand(V, brain + (blockIdx.x * (nr_synapsis / 8)), gridDim.x, board, board_size);

    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("a\n");
        result = !!is_set(V[(128/8)-1], 7);
    }
}
*/

__device__ void cudump(unsigned char *arr, int n)
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

__global__ void cunand(unsigned char *V, unsigned char *brain, int size,
unsigned char *board, int board_size, int i)
{
    int index = threadIdx.x;
    __shared__ int found;

    found = 0;

    __syncthreads();
    if (index < size && isolate_bit(V[index / 8], index % 8) &
            isolate_bit(brain[index / 8], index % 8)) {
        found = 1;
        return;
    }

    __syncthreads();
    if (found)
        return;

    index <<= 1;
    if (isolate_bit(board[index / 8], index % 8) &&
            isolate_bit(brain[(index + size) / 8], (index + size) % 8)) {
        found = 1;
        return;
    }

    index++;
    if (isolate_bit(board[index / 8], index % 8) &&
            isolate_bit(brain[(index + size) / 8], (index + size) % 8)) {
        found = 1;
        return;
    }
    if (!found)
        set_bit(V[i / 8], i % 8);
}

int cuscore(unsigned char *brain, unsigned char *board,
        int board_size, int nr_synapsis, unsigned char *V, int nr_ports, int la)
{
    int i, ret;
    dim3 threads_pr_block(1024);
    unsigned char tmpV[128/8];

    cudaMemset(V, 0, nr_ports/8);
    for (i = 0; i < nr_ports; ++i) {
        cunand<<<1, threads_pr_block>>>(V, brain + (i * (nr_synapsis / 8)),
                nr_ports, board, board_size, i);
    }

    ret = cudaMemcpy(tmpV, V, nr_ports / 8, cudaMemcpyDeviceToHost);
    switch (ret) {
        case cudaSuccess:
            break;
        case cudaErrorInvalidValue:
            printf("invalid value!\n");
            break;
        case cudaErrorInvalidDevicePointer:
            printf("invalid device pointer!\n");
            break;
        case cudaErrorInvalidMemcpyDirection:
            printf("invalid memcpy direction!\n");
            break;
        case cudaErrorInvalidSymbol:
            printf("invalid symbol!\n");
            break;
        default:
            printf("unknown error!\n");
            break;
    }

    if (la) {
        printf("cuda V:\n");
        dump(tmpV, sizeof(tmpV));
    }

    return !!is_set(tmpV[(128/8)-1], 7);
}

static int cu_get_best_move(AI_instance_t *ai, board_t *board)
{
    unsigned char *cubrain, *cuboard, *cuV;
    int x, y, ts, ret;

    int i, count, moveret;
    float cumdist[board->moves_count], fcount;
    int scores[board->moves_count];

    mem_2d_get_dims((void **)ai->brain, &x, &y, &ts);
    cudaMalloc(&cubrain, x * y * ts);
    ret = cudaMemcpy(cubrain, &ai->brain[0][0], x * y * ts, cudaMemcpyHostToDevice);
    switch (ret) {
        case cudaSuccess:
            break;
        case cudaErrorInvalidValue:
            printf("invalid value!\n");
            break;
        case cudaErrorInvalidDevicePointer:
            printf("invalid device pointer!\n");
            break;
        case cudaErrorInvalidMemcpyDirection:
            printf("invalid memcpy direction!\n");
            break;
        case cudaErrorInvalidSymbol:
            printf("invalid symbol!\n");
            break;
        default:
            printf("unknown error!\n");
            break;
    }

    cudaMalloc(&cuV, ai->nr_ports/8);

    cudaMalloc(&cuboard, ai->board_size / 8);

    memcpy(&board->board[64], &board->board[0], 64 * sizeof(piece_t));
    for (i = count = 0; i < board->moves_count; i = count++) {
        moveret = move(board, i);

        /* move returns 1 on success , -1 if stalemate, 0 if i > board->moves_count */
        if (moveret == 1) {
            ret = cudaMemcpy(cuboard, board->board, ai->board_size / 8, cudaMemcpyHostToDevice);
            switch (ret) {
                case cudaSuccess:
                    break;
                case cudaErrorInvalidValue:
                    printf("invalid value!\n");
                    break;
                case cudaErrorInvalidDevicePointer:
                    printf("invalid device pointer!\n");
                    break;
                case cudaErrorInvalidMemcpyDirection:
                    printf("invalid memcpy direction!\n");
                    break;
                case cudaErrorInvalidSymbol:
                    printf("invalid symbol!\n");
                    break;
                default:
                    printf("unknown error!\n");
                    break;
            }

            scores[i] = cuscore(cubrain, cuboard,
                    ai->board_size, ai->nr_synapsis, cuV, ai->nr_ports, i == 9);

            undo_move(board, i);
            continue;
        } else if (!moveret)
            break;
        else if (moveret == -1)
            return -1;
    }

    fcount = 0;
    for (i = 0; i < board->moves_count; i++) {
        fcount += scores[i];
        cumdist[i] = fcount;
    }
    x = 2.5 * cumdist[board->moves_count - 1];

    printf("cuda score:\n");
    dump(scores, sizeof(scores));


    return bisect(cumdist, x, board->moves_count);

}

static int _get_best_move(AI_instance_t *ai, board_t *board)
{
    int i, count, moveret;
    float cumdist[board->moves_count], fcount, x;
    int scores[board->moves_count];

    memcpy(&board->board[64], &board->board[0], 64 * sizeof(piece_t));
    for (i = count = 0; i < board->moves_count; i = count++) {
        moveret = move(board, i);

        /* move returns 1 on success */
        if (moveret == 1) {
            scores[i] = score(ai, board->board, i == 9);
            undo_move(board, i);
            continue;
        }

        /* move returns -1 if stalemate, 0 if i > board->moves_count */
        if (!moveret)
            break;
        else if (moveret == -1)
            return -1;
    }

    fcount = 0;
    for (i = 0; i < board->moves_count; i++) {
        fcount += scores[i];
        cumdist[i] = fcount;
    }
    x = 2.5 * cumdist[board->moves_count - 1];

    printf("C score:\n");
    dump(scores, sizeof(scores));
    return bisect(cumdist, x, board->moves_count);
}

int main(int argc, char **argv)
{
    int cuda, c, count = 0;

    AI_instance_t *ai = ai_new(10, 1);
    //AI_instance_t *ai = ai_new();
    board_t *board = new_board(NULL);

    while (count < 10000) {
        generate_all_moves(board);

        //if (argc > 2)
        cuda = cu_get_best_move(ai, board);
        //else
        c = _get_best_move(ai, board);

        printf("cuda: %d\n", cuda);
        printf("C   : %d\n", c);

        do_random_move(board);
        c = getchar();

        if (c != '\n')
            break;
        ++count;
    }

    return 0;
}
