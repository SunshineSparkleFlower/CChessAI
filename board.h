#ifndef __BOARD_H
#define __BOARD_H

#include <stdint.h>

typedef uint16_t piece_t;

typedef struct coord {
    int8_t y :4;
    int8_t x :4;
} coord_t;

typedef struct move {
    coord_t frm, to;
} move_t;

#include "common.h"
struct bitboard {
    u64 pieces, apieces;
    u64 pawns;
    u64 rooks;
    u64 knights;
    u64 bishops;
    u64 queens;
    u64 king;
};

typedef struct board {
    piece_t _board[8 * 8 * 2];
    piece_t *board_2d[8];
    piece_t *board; // only for backwards compatability. points to _board
    piece_t *cu_board;
    struct bitboard white_pieces, black_pieces;
    // used in case an illegal move is made
    struct {
        u64 *capture_board, *move_board;
        u64 capture_mask;
        int promotion;
        piece_t piece;
    } backup;

    struct move moves[20*16];
    int moves_count;
    int turn;
    int is_check;
} board_t;

#define DEFAULT_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"

extern void set_board(board_t *board, const char *_fen);
extern board_t *new_board(const char *_fen);
extern void free_board(board_t *b);
extern void generate_all_moves(board_t *b);
extern int is_stalemate(board_t *b);
extern int is_check(board_t *board);
extern int is_checkmate(board_t *b);
extern void swapturn(board_t *b);
extern int undo_move(board_t *b, int n);
extern int move(board_t *b, int n);
extern int do_move(board_t *b, int n);
extern char *get_fen(board_t *board);
extern void print_board(piece_t *board);
extern void print_move(board_t *board, int n);
extern void print_legal_moves(board_t *board);

extern void init_magicmoves(void);

#endif
