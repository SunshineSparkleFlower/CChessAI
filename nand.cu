#include <stdio.h>
#include <unistd.h>

#include "common.h"
#include "AI.h"
#include "board.h"
#include "nand.h"
__device__ int result;

__device__ int cu_nand(int *a, int *b, int size, piece_t *board, int board_size) {
    int i;
    int ret = 1;
    for (i = 0; i < size/32; i++) {
        if (a[i] & b[i]) {
            ret =  0;
        }
    }

    for (i = 0; i < board_size/32; i++) {
        if (board[i] &b[i + size/32]) {
            ret = 0;
        }
    }
    return ret;

}

__global__ void cu_eval_curcuit(int* brainpart_value, int *M, int nr_ports, piece_t *board, int board_size) {
    __shared__ piece_t s_bord[128];
        if (threadIdx.x == 1 && blockIdx.x == 1) {
        for(int i = threadIdx.x; i  <128; i ++) {
            s_bord[i] = board[i];
        }
    }
    __syncthreads();
    const int max_prts = 128;
    int V[max_prts];

    int brain_idx = blockIdx.x;
    int i;
    int size_port = (nr_ports + board_size)/32;
    int brain = brain_idx*(nr_ports*size_port);
    

    for (i = 0; i < nr_ports; i++) {
        //get the value of port i using a offset
        if (cu_nand(V, M+brain+i*size_port, nr_ports, s_bord, board_size))
            SetBit(V, i);
    }
    brainpart_value[brain_idx] = !!TestBit(V,(nr_ports-1));

}

int cu_score(AI_instance_t *ai, piece_t *board) {
    dim3 blocks(ai->nr_brain_parts, 1);
    dim3 grids(1, 1);
    //piece_t *cu_board;
    int *cu_brainpart_value;
    int *brainpart_value;

    //allocate memory for CUDA
    //cudaMalloc(&cu_board, sizeof(piece_t)*64);
    cudaMalloc(&cu_brainpart_value, ai->nr_brain_parts * sizeof (int));
    brainpart_value = (int *) malloc(ai->nr_brain_parts * sizeof (int));

   
    cudaMemcpy(ai->cu_board, board, sizeof(piece_t)*128, cudaMemcpyHostToDevice);

    cu_eval_curcuit <<<1, ai->nr_brain_parts >>>(cu_brainpart_value, ai->cu_brain, ai->nr_ports, ai->cu_board, ai->board_size);
    //wait for all the brain parts to finish
    if(cudaDeviceSynchronize())
        printf("synch error\n");
    
    //get the results from each brain part
    cudaMemcpy(brainpart_value, cu_brainpart_value, ai->nr_brain_parts * sizeof (int), cudaMemcpyDeviceToHost);

    int score_sum = 0;
    int i;
    for (i = 0; i < ai->nr_brain_parts; i++) {
        score_sum += brainpart_value[i];
    }
    //free the allocated memory
    cudaFree(cu_brainpart_value);
    free(brainpart_value);
    return score_sum;
}
