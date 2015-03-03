#ifndef __BITBOARD_H
#define __BITBOARD_H

#include <stdint.h>

enum coords {
    H1, G1, F1, E1, D1, C1, B1, A1,
    H2, G2, F2, E2, D2, C2, B2, A2,
    H3, G3, F3, E3, D3, C3, B3, A3,
    H4, G4, F4, E4, D4, C4, B4, A4,
    H5, G5, F5, E5, D5, C5, B5, A5,
    H6, G6, F6, E6, D6, C6, B6, A6,
    H7, G7, F7, E7, D7, C7, B7, A7,
    H8, G8, F8, E8, D8, C8, B8, A8,
};

#define bb_col_to_AI_col(col) (~(col) & 0x7)
#define bb_coord_to_index(row, col) ((row) * 8 + (col))

#define AI_col_to_bb_col(col) (~(col) & 0x7)
#define AI_coord_to_index(row, col) ((row) * 8 + AI_col_to_bb_col(col))

#include "board.h"
extern void init_bitboards(char *_fen, board_t *board);
extern void bb_print(u64 b);
extern int bb_can_attack(u64 moves, int pos);
extern int bb_calculate_check(board_t *board);
extern void bb_generate_all_legal_moves(board_t *board);
extern int bb_do_move(board_t *b, int index);
extern int bb_undo_move(board_t *b, int index);
extern int bb_do_actual_move(board_t *board, struct move *m);
extern void generate_king_moves_only(board_t *board);
extern void generate_queen_moves_only(board_t *board);
extern void generate_rook_moves_only(board_t *board);
extern void generate_bishop_moves_only(board_t *board);
extern void generate_knight_moves_only(board_t *board);
extern void generate_pawn_moves_only(board_t *board);
#endif
