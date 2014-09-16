#ifndef __RULES_H /* start of include guard */
#define __RULES_H

#include "common.h"

extern int get_legal_moves(piece_t *board[8], coord_t *from);
extern legal_moves_t *get_all_legal_moves(piece_t *board[8]);
extern int is_check(int player_color, piece_t *board[8]);

#endif /* end of include guard: __RULES_H */
