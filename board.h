#ifndef __BOARD_H
#define __BOARD_H

#include "common.h"
#include <stdint.h>

extern board_t *new_board(char *_fen);
extern void free_board(board_t *b);
extern int have_lost(board_t *b);
extern void swapturn(board_t *b);
extern void move(board_t *b, int n);
extern char *get_fen(board_t *board);
extern void print_board(piece_t *board);
extern void print_legal_moves(board_t *board);

#endif
