#ifndef __BOARD_H
#define __BOARD_H

#include <stdint.h>
#include "uci.h"

typedef uint16_t piece_t;

typedef struct coord {
    int8_t y :4;
    int8_t x :4;
} coord_t;

#include "common.h"
typedef struct move {
    u8 en_passant :1; // 1 if its an en passant move. 0 otherwise
    u8 promotion; // 'q' for queen, 'k' for knight, 'b' for bishop, 'r' for rook
    coord_t frm, to;
} move_t;

struct bitboard {
    u8 king_has_moved :1;
    u8 long_rook_moved :1;
    u8 short_rook_moved :1;
    // must be updated on each move.
    // 0-7 indicates the column of double move.
    // > 7 indicates that no double move was done
    u8 double_pawn_move :4;

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
    struct bitboard white_pieces, black_pieces;
    // used in case an illegal move is made
    struct {
        u64 *capture_board, *move_board;
        u64 capture_mask;
        int promotion;
        piece_t piece;

        u8 castling :2; // 0 = no castling, 1 short, 2 long
        u8 king_had_moved :1;
        u8 long_rook_had_moved :1;
        u8 short_rook_had_moved :1;
    } backup;

    struct move moves[20*16];
    int moves_count;
    int turn;
    int is_check;
} board_t;

#define DEFAULT_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

extern board_t *new_board(char *_fen);
extern void free_board(board_t *b);
extern void generate_all_moves(board_t *b);
extern int is_stalemate(board_t *b);
extern int is_check(board_t *board);
extern int is_checkmate(board_t *b);
extern void swapturn(board_t *b);
extern int undo_move(board_t *b, int n);
extern int move(board_t *b, int n);
extern int do_move(board_t *b, int n);
extern int do_actual_move(board_t *b, struct move *m, struct uci *iface);
extern char *get_fen(board_t *board);
extern void print_board(piece_t *board);
extern void print_move(board_t *board, int n);
extern void print_legal_moves(board_t *board);
extern const char *piece_to_str(piece_t p);
extern int do_uci_move(board_t *board, struct uci *iface);
extern void register_move_to_uci(struct move *m, struct uci *iface);
extern void board_consistency_check(board_t *board);

extern void init_magicmoves(void);

#endif
