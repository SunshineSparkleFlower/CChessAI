#ifndef __RULES_H /* start of include guard */
#define __RULES_H

#include "common.h"

extern int get_legal_moves(board_t *board_struct, coord_t *from);
extern board_t *get_all_legal_moves(board_t *board_struct);
extern int is_check(int player_color, board_t *board_struct);
extern void print_board(piece_t *board);

#endif /* end of include guard: __RULES_H */
