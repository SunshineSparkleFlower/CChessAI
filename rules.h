#ifndef __RULES_H /* start of include guard */
#define __RULES_H

#include "common.h"

extern int get_legal_moves(piece_t *board, coord_t *from);
extern legal_moves_t *get_all_legal_moves(piece_t *board);
extern int is_check(int player_color, piece_t *board);
extern void print_board(piece_t *board);

#endif /* end of include guard: __RULES_H */
