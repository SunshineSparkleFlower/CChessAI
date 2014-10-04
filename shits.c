#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <sys/time.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include "bitboard.h"

typedef int bool;
#define false 0
#define true 1

typedef int direction_t;
typedef enum {
    NONE_FLAG=0, WP_FLAG=1<<0, BP_FLAG=1<<1, N_FLAG=1<<2,
    B_FLAG=1<<3, R_FLAG=1<<4, Q_FLAG=1<<5, K_FLAG=1<<6
} piece_flag_tag_t;
typedef int piece_flag_t;
typedef struct {
    piece_flag_t possible_attackers;
    direction_t relative_direction;
} attack_data_t;
attack_data_t board_attack_data_storage[256];
const attack_data_t* board_attack_data = board_attack_data_storage + 128;

const piece_flag_t piece_flags[] = {
    0, WP_FLAG, N_FLAG, B_FLAG, R_FLAG, Q_FLAG, K_FLAG, 0,
    0, BP_FLAG, N_FLAG, B_FLAG, R_FLAG, Q_FLAG, K_FLAG, 0, 0
};
#define get_attack_data(from, to)   board_attack_data[(from)-(to)]
#define distance(from,to)           distance_data[(from)-(to)]
#define direction(from, to) \
    get_attack_data((from),(to)).relative_direction
#define possible_attack(from, to, piece) \
    ((get_attack_data((from),(to)).possible_attackers & \
     piece_flags[(piece)]) != 0)
#define get_attack_data(from, to)   board_attack_data[(from)-(to)]
#define distance(from,to)           distance_data[(from)-(to)]
#define direction(from, to) \
    get_attack_data((from),(to)).relative_direction
#define possible_attack(from, to, piece) \
    ((get_attack_data((from),(to)).possible_attackers & \
     piece_flags[(piece)]) != 0)

typedef enum {
    WHITE=0, BLACK=1, INVALID_COLOR=2
} color_t;

typedef enum {
    EMPTY=0, WP=1, WN=2, WB=3, WR=4, WQ=5, WK=6,
    BP=9, BN=10, BB=11, BR=12, BQ=13, BK=14,
    OUT_OF_BOUNDS=16
} piece_t;

typedef enum {
    NONE=0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
} piece_type_t;

typedef struct {
    int midgame;
    int endgame;
} score_t;

typedef enum {
    A1=0x00, B1=0x01, C1=0x02, D1=0x03, E1=0x04, F1=0x05, G1=0x06, H1=0x07,
    A2=0x10, B2=0x11, C2=0x12, D2=0x13, E2=0x14, F2=0x15, G2=0x16, H2=0x17,
    A3=0x20, B3=0x21, C3=0x22, D3=0x23, E3=0x24, F3=0x25, G3=0x26, H3=0x27,
    A4=0x30, B4=0x31, C4=0x32, D4=0x33, E4=0x34, F4=0x35, G4=0x36, H4=0x37,
    A5=0x40, B5=0x41, C5=0x42, D5=0x43, E5=0x44, F5=0x45, G5=0x46, H5=0x47,
    A6=0x50, B6=0x51, C6=0x52, D6=0x53, E6=0x54, F6=0x55, G6=0x56, H6=0x57,
    A7=0x60, B7=0x61, C7=0x62, D7=0x63, E7=0x64, F7=0x65, G7=0x66, H7=0x67,
    A8=0x70, B8=0x71, C8=0x72, D8=0x73, E8=0x74, F8=0x75, G8=0x76, H8=0x77,
    INVALID_SQUARE=0x4b // just some square from the middle of the invalid part
} square_tag_t;
typedef int square_t;
typedef int32_t move_t;
typedef uint8_t castle_rights_t;
typedef uint64_t hashkey_t;

typedef struct {
    piece_t _board_storage[256];        // 16x16 padded board
    piece_t* board;                     // 0x88 board in middle 128 slots
    int piece_index[128];               // index of each piece in pieces
    square_t pieces[2][32];
    square_t pawns[2][16];
    int num_pieces[2];
    int num_pawns[2];
    int piece_count[16];
    color_t side_to_move;
    move_t prev_move;
    square_t ep_square;
    int fifty_move_counter;
    int ply;
    int material_eval[2];
    score_t piece_square_eval[2];
    castle_rights_t castle_rights;
    uint8_t is_check;
    square_t check_square;
    hashkey_t hash;
    hashkey_t pawn_hash;
    hashkey_t material_hash;
#define HASH_HISTORY_LENGTH  2048
    hashkey_t hash_history[HASH_HISTORY_LENGTH];
} position_t;

position_t fuckeverything;
move_t moves[256];

#define warn(msg)   do { \
    FILE* log; \
    log = fopen("daydreamer.log", "a"); \
    printf("%s:%u: %s\n", __FILE__, __LINE__, msg); \
    fprintf(log, "\n%s %s\n", __DATE__, __TIME__); \
    fprintf(log, "%s:%u: %s\n", __FILE__, __LINE__, msg); \
    fclose(log); \
} while (0)

#define square_file(square)         ((square) & 0x0f)
#define direction(from, to) \
    get_attack_data((from),(to)).relative_direction
#define create_piece(color, type)       (((color) << 3) | (type))
#define index_to_square(index)      ((index)+((index) & ~0x07))
#define square_to_index(square)     ((square)+((square) & 0x07))>>1
#define piece_color(piece)              ((piece) >> 3)
#define piece_type(piece)               ((piece) & 0x07)
#define piece_hash(p,sq) \
    piece_random[piece_color(p)][piece_type(p)][square_to_index(sq)]
#define piece_slide_type(piece)     (sliding_piece_types[piece])
#define check_board_validity(x)                 ((void)0)
#define check_pseudo_move_legality(x,y)         ((void)0)
#define create_move_enpassant(from, to, piece, capture) \
    ((from) | ((to) << 8) | ((piece) << 16) | \
     ((capture) << 20) | ENPASSANT_FLAG)
#define create_move_promote(from, to, piece, capture, promote) \
    ((from) | ((to) << 8) | ((piece) << 16) | \
     ((capture) << 20) | ((promote) << 24))
#define create_move_castle(from, to, piece) \
    ((from) | ((to) << 8) | ((piece) << 16) | CASTLE_FLAG)

/*
 * Each move is a 4-byte quantity that encodes source and destination
 * square, the moved piece, any captured piece, the promotion value (if any),
 * and flags to indicate en-passant capture and castling. The bit layout of
 * a move is as follows:
 *
 * 12345678  12345678  1234     5678     1234          5    6       78
 * FROM      TO        PIECE    CAPTURE  PROMOTE       EP   CASTLE  UNUSED
 * square_t  square_t  piece_t  piece_t  piece_type_t  bit  bit
 */
typedef int32_t move_t;
#define NO_MOVE                     0
#define NULL_MOVE                   0xffff
#define ENPASSANT_FLAG              1<<29
#define CASTLE_FLAG                 1<<30
#define get_move_from(move)         ((move) & 0xff)
#define get_move_to(move)           (((move) >> 8) & 0xff)
#define get_move_piece(move)        (((move) >> 16) & 0x0f)
#define get_move_piece_type(move) \
    ((piece_type_t)piece_type(get_move_piece(move)))
#define get_move_piece_color(move)  ((color_t)piece_color(get_move_piece(move)))
#define get_move_capture(move)      (((move) >> 20) & 0x0f)
#define get_move_capture_type(move) piece_type(get_move_capture(move))
#define get_move_promote(move)      (((move) >> 24) & 0x0f)
#define is_move_enpassant(move)     (((move) & ENPASSANT_FLAG) != 0)
#define is_move_castle(move)        (((move) & CASTLE_FLAG) != 0)
#define is_move_castle_long(move) \
    (is_move_castle(move) && (square_file(get_move_to(move)) == FILE_C))
#define is_move_castle_short(move) \
    (is_move_castle(move) && (square_file(get_move_to(move)) == FILE_G))
#define create_move(from, to, piece, capture) \
    ((from) | ((to) << 8) | ((piece) << 16) | ((capture) << 20))
#define create_move_promote(from, to, piece, capture, promote) \
    ((from) | ((to) << 8) | ((piece) << 16) | \
     ((capture) << 20) | ((promote) << 24))
#define create_move_castle(from, to, piece) \
    ((from) | ((to) << 8) | ((piece) << 16) | CASTLE_FLAG)
#define create_move_enpassant(from, to, piece, capture) \
    ((from) | ((to) << 8) | ((piece) << 16) | \
     ((capture) << 20) | ENPASSANT_FLAG)
#define piece_type(piece)               ((piece) & 0x07)
#define piece_is_type(piece, type)      (piece_type((piece)) == (type))
#define piece_color(piece)              ((piece) >> 3)
#define piece_is_color(piece, color)    (piece_color((piece)) == (color))
#define create_piece(color, type)       (((color) << 3) | (type))
#define piece_colors_match(p1, p2)      (((p1) >> 3) == ((p2) >> 3))
#define piece_colors_differ(p1, p2)     (((p1) >> 3) != ((p2) >> 3))
#define can_capture(p1, p2)             ((((p1) >> 3)^1) == ((p2) >> 3))
#define flip_piece(p)                   (flip_piece[p])

typedef enum {
    NO_SLIDE=0, DIAGONAL, STRAIGHT, BOTH
} slide_t;
typedef int direction_t;

typedef enum {
    SSW=-33, SSE=-31,
    WSW=-18, SW=-17, S=-16, SE=-15, ESE=-14,
    W=-1, STATIONARY=0, E=1,
    WNW=14, NW=15, N=16, NE=17, ENE=18,
    NNW=31, NNE=33
} direction_tag_t;

const direction_t piece_deltas[17][16] = {
    // White Pieces
    {0},                                                    // Null
    {NW, NE, 0},                                            // Pawn
    {SSW, SSE, WSW, ESE, WNW, ENE, NNW, NNE, 0},            // Knight
    {SW, SE, NW, NE, 0},                                    // Bishop
    {S, W, E, N, 0},                                        // Rook
    {SW, S, SE, W, E, NW, N, NE, 0},                        // Queen
    {SW, S, SE, W, E, NW, N, NE, 0},                        // King
    {0}, {0},                                               // Null
    // Black Pieces
    {SE, SW, 0},                                            // Pawn
    {SSW, SSE, WSW, ESE, WNW, ENE, NNW, NNE, 0},            // Knight
    {SW, SE, NW, NE, 0},                                    // Bishop
    {S, W, E, N, 0},                                        // Rook
    {SW, S, SE, W, E, NW, N, NE, 0},                        // Queen
    {SW, S, SE, W, E, NW, N, NE, 0},                        // King
    {0}, {0}                                                // Null
};

const slide_t sliding_piece_types[] = {
    0, 0, 0, DIAGONAL, STRAIGHT, BOTH, 0, 0,
    0, 0, 0, DIAGONAL, STRAIGHT, BOTH, 0, 0, 0
};
/*
 * Is |sq| being directly attacked by any pieces on |side|? Works on both
 * occupied and unoccupied squares.
 */
bool is_square_attacked(const position_t* pos, square_t sq, color_t side)
{
    // For every opposing piece, look up the attack data for its square.
    // Special-case pawns for efficiency.
    piece_t opp_pawn = create_piece(side, PAWN);
    if (pos->board[sq - piece_deltas[opp_pawn][0]] == opp_pawn) return true;
    if (pos->board[sq - piece_deltas[opp_pawn][1]] == opp_pawn) return true;

    square_t from;
    for (const square_t* pfrom = &pos->pieces[side][0];
            (from = *pfrom) != INVALID_SQUARE;
            ++pfrom) {
        piece_t p = pos->board[from];
#define piece_slide_type(piece)     (sliding_piece_types[piece])
        if (possible_attack(from, sq, p)) {
            if (piece_slide_type(p) == NO_SLIDE) return true;
            direction_t att_dir = direction(from, sq);
            while (from != sq) {
                from += att_dir;
                if (from == sq) return true;
                if (pos->board[from] != EMPTY) break;
            }
        }
    }
    return false;
}

/*
 * Is the piece on |from| pinned, and if so what's the direction to the king?
 */
direction_t pin_direction(const position_t* pos,
        square_t from,
        square_t king_sq)
{
    direction_t pin_dir = 0;
    if (!possible_attack(from, king_sq, WQ)) return 0;
    direction_t king_dir = direction(from, king_sq);
    square_t sq;
    for (sq = from + king_dir; pos->board[sq] == EMPTY; sq += king_dir) {}
    if (sq == king_sq) {
        // Nothing between us and the king. Is there anything
        // behind us that's doing the pinning?
        for (sq = from - king_dir; pos->board[sq] == EMPTY; sq -= king_dir) {}
        if (can_capture(pos->board[sq], pos->board[from]) &&
                possible_attack(from, king_sq, pos->board[sq]) &&
                (piece_slide_type(pos->board[sq]) != NONE)) {
            pin_dir = king_dir;
        }
    }
    return pin_dir;
}
typedef struct {
    uint8_t is_check;
    square_t check_square;
    move_t prev_move;
    square_t ep_square;
    int fifty_move_counter;
    castle_rights_t castle_rights;
    hashkey_t hash;
} undo_info_t;

#define square_rank(square)         ((square) >> 4)
#define square_file(square)         ((square) & 0x0f)
#define square_color(square)        (((square) ^ ((square) >> 4) ^ 1) & 1)
#define create_square(file, rank)   (((rank) << 4) | (file))
#define valid_board_index(idx)      !((idx) & 0x88)
#define flip_square(square)         ((square) ^ 0x70)
#define mirror_rank(square)         ((square) ^ 0x70)
#define mirror_file(square)         ((square) ^ 0x07)
#define square_to_index(square)     ((square)+((square) & 0x07))>>1
#define index_to_square(index)      ((index)+((index) & ~0x07))

typedef uint8_t castle_rights_t;
#define WHITE_OO                        0x01
#define BLACK_OO                        0x01 << 1
#define WHITE_OOO                       0x01 << 2
#define BLACK_OOO                       0x01 << 3
#define CASTLE_ALL                      (WHITE_OO | BLACK_OO | \
                                            WHITE_OOO | BLACK_OOO)
#define CASTLE_NONE                     0
#define has_oo_rights(pos, side)        ((pos)->castle_rights & \
                                            (WHITE_OO<<(side)))
#define has_ooo_rights(pos, side)       ((pos)->castle_rights & \
                                            (WHITE_OOO<<(side)))
#define can_castle(pos, side)           (has_oo_rights(pos,side) || \
                                            has_ooo_rights(pos,side))
#define add_oo_rights(pos, side)        ((pos)->castle_rights |= \
                                            (WHITE_OO<<(side)))
#define add_ooo_rights(pos, side)       ((pos)->castle_rights |= \
                                            (WHITE_OOO<<(side)))
#define remove_oo_rights(pos, side)     ((pos)->castle_rights &= \
                                            ~(WHITE_OO<<(side)))
#define remove_ooo_rights(pos, side)    ((pos)->castle_rights &= \
                                            ~(WHITE_OOO<<(side)))
const hashkey_t piece_random[2][7][64];
const hashkey_t castle_random[2][2][2];
const hashkey_t enpassant_random[64];
#define piece_hash(p,sq) \
    piece_random[piece_color(p)][piece_type(p)][square_to_index(sq)]
#define ep_hash(pos) \
    enpassant_random[square_to_index((pos)->ep_square)]
#define castle_hash(pos) \
    (castle_random[has_oo_rights(pos, WHITE) ? 1 : 0][0][0] ^ \
    castle_random[has_ooo_rights(pos, WHITE) ? 1 : 0][0][1] ^ \
    castle_random[has_oo_rights(pos, BLACK) ? 1 : 0][1][0] ^ \
    castle_random[has_ooo_rights(pos, BLACK) ? 1 : 0][1][1])
#define side_hash(pos)  ((pos)->side_to_move * 0x823a67c5f88337e7ull)
#define material_hash(p,count) \
    piece_random[piece_color(p)][piece_type(p)][count]

/*
 * Modify |pos| by performing the given |move|. The information needed to undo
 * this move is preserved in |undo|.
 */
typedef enum {
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NONE
} rank_tag_t;
typedef int rank_t;
const rank_t relative_rank[2][8] = {
    {RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8},
    {RANK_8, RANK_7, RANK_6, RANK_5, RANK_4, RANK_3, RANK_2, RANK_1}
};
const direction_t pawn_push[] = {N, S};
square_t king_rook_home = H1;
square_t queen_rook_home = A1;
square_t king_home = E1;
#define MAX_PHASE           24

#define PAWN_VAL         85
#define KNIGHT_VAL       350
#define BISHOP_VAL       350
#define ROOK_VAL         500
#define QUEEN_VAL        1000
#define KING_VAL         20000
#define EG_PAWN_VAL      115
#define EG_KNIGHT_VAL    400
#define EG_BISHOP_VAL    400
#define EG_ROOK_VAL      650
#define EG_QUEEN_VAL     1200
#define EG_KING_VAL      20000
#define WON_ENDGAME     (2*EG_QUEEN_VAL)
const int material_values[] = {
    0, PAWN_VAL, KNIGHT_VAL, BISHOP_VAL, ROOK_VAL, QUEEN_VAL, KING_VAL, 0,
    0, PAWN_VAL, KNIGHT_VAL, BISHOP_VAL, ROOK_VAL, QUEEN_VAL, KING_VAL, 0, 0
};
int endgame_piece_square_values[BK+1][0x80] = {
    {}, {}, {}, {}, {}, {}, {}, {}, {}, // empties to get indexing right
{ // pawn
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
  6,  4,  2,  0,  0,  2,  4,  6, 0, 0, 0, 0, 0, 0, 0, 0,
  4,  2,  0, -2, -2,  0,  2,  4, 0, 0, 0, 0, 0, 0, 0, 0,
  3,  1, -1, -3, -3, -1,  1,  3, 0, 0, 0, 0, 0, 0, 0, 0,
  2,  0, -2, -4, -4, -2,  0,  2, 0, 0, 0, 0, 0, 0, 0, 0,
  1, -1, -3, -5, -5, -3, -1,  1, 0, 0, 0, 0, 0, 0, 0, 0,
  1, -1, -3, -5, -5, -3, -1,  1, 0, 0, 0, 0, 0, 0, 0, 0,
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // knight
 -12,  -7,  -2,   1,   1,  -2,  -7, -12, 0, 0, 0, 0, 0, 0, 0, 0,
  -5,   2,   6,   8,   8,   6,   2,  -5, 0, 0, 0, 0, 0, 0, 0, 0,
   0,   6,  11,  13,  13,  11,   6,   0, 0, 0, 0, 0, 0, 0, 0, 0,
  -1,   4,   9,  13,  13,   9,   4,  -1, 0, 0, 0, 0, 0, 0, 0, 0,
  -3,   2,   7,  11,  11,   7,   2,  -3, 0, 0, 0, 0, 0, 0, 0, 0,
  -7,  -1,   4,   6,   6,   4,  -1,  -7, 0, 0, 0, 0, 0, 0, 0, 0,
 -12,  -5,  -1,   1,   1,  -1,  -5, -12, 0, 0, 0, 0, 0, 0, 0, 0,
 -19, -14,  -9,  -6,  -6,  -9, -14, -19, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // bishop
  0, -1, -2, -2, -2, -2, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0,
 -1,  1,  0,  0,  0,  0,  1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
 -2,  0,  3,  2,  2,  3,  0, -2, 0, 0, 0, 0, 0, 0, 0, 0,
 -2,  0,  2,  5,  5,  2,  0, -2, 0, 0, 0, 0, 0, 0, 0, 0,
 -2,  0,  2,  5,  5,  2,  0, -2, 0, 0, 0, 0, 0, 0, 0, 0,
 -2,  0,  3,  2,  2,  3,  0, -2, 0, 0, 0, 0, 0, 0, 0, 0,
 -1,  1,  0,  0,  0,  0,  1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
  0, -1, -2, -2, -2, -2, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // rook
 -2, -2, -2, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0,
  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, 0, 0, 0, 0, 0,
  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, 0, 0, 0, 0, 0,
  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, 0, 0, 0, 0, 0,
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // queen
-11, -6, -4, -3, -3, -4, -6,-11, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -1,  1,  2,  2,  1, -1, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -4,  1,  4,  6,  6,  4,  1, -4, 0, 0, 0, 0, 0, 0, 0, 0,
 -3,  2,  6,  9,  9,  6,  2, -3, 0, 0, 0, 0, 0, 0, 0, 0,
 -3,  2,  6,  9,  9,  6,  2, -3, 0, 0, 0, 0, 0, 0, 0, 0,
 -4,  1,  4,  6,  6,  4,  1, -4, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -1,  1,  2,  2,  1, -1, -6, 0, 0, 0, 0, 0, 0, 0, 0,
-11, -6, -4, -3, -3, -4, -6,-11, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // king
-42,-19, -3,  3,  3, -3,-19,-42, 0, 0, 0, 0, 0, 0, 0, 0,
-24,  1, 13, 19, 19, 13,  1,-24, 0, 0, 0, 0, 0, 0, 0, 0,
-13,  8, 23, 29, 29, 23,  8,-13, 0, 0, 0, 0, 0, 0, 0, 0,
 -7, 14, 29, 38, 38, 29, 14, -7, 0, 0, 0, 0, 0, 0, 0, 0,
-12,  9, 24, 33, 33, 24,  9,-12, 0, 0, 0, 0, 0, 0, 0, 0,
-18,  3, 18, 24, 24, 18,  3,-18, 0, 0, 0, 0, 0, 0, 0, 0,
-29, -4,  8, 14, 14,  8, -4,-29, 0, 0, 0, 0, 0, 0, 0, 0,
-62,-39,-23,-17,-17,-23,-39,-62, 0, 0, 0, 0, 0, 0, 0, 0, },
    // mirror piece tables are filled in during init_eval
};
int piece_square_values[BK+1][0x80] = {
    {}, {}, {}, {}, {}, {}, {}, {}, {}, // empties to get indexing right
{ // pawn
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
-10,  2,  8, 15, 15,  8,  2,-10, 0, 0, 0, 0, 0, 0, 0, 0,
-11,  1,  7, 13, 13,  7,  1,-11, 0, 0, 0, 0, 0, 0, 0, 0,
-12,  0,  5, 12, 12,  5,  0,-12, 0, 0, 0, 0, 0, 0, 0, 0,
-14, -2,  3, 10, 10,  3, -2,-14, 0, 0, 0, 0, 0, 0, 0, 0,
-15, -4,  2,  9,  9,  2, -4,-15, 0, 0, 0, 0, 0, 0, 0, 0,
-16, -5,  1,  8,  8,  1, -5,-16, 0, 0, 0, 0, 0, 0, 0, 0,
  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // knight
 -82, -16,  -5,   0,   0,  -5, -16, -82, 0, 0, 0, 0, 0, 0, 0, 0,
 -11,   6,  17,  21,  21,  17,   6, -11, 0, 0, 0, 0, 0, 0, 0, 0,
  -1,  15,  26,  30,  30,  26,  15,  -1, 0, 0, 0, 0, 0, 0, 0, 0,
   0,  17,  28,  32,  32,  28,  17,   0, 0, 0, 0, 0, 0, 0, 0, 0,
  -6,  11,  22,  26,  26,  22,  11,  -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -15,   1,  13,  17,  17,  13,   1, -15, 0, 0, 0, 0, 0, 0, 0, 0,
 -31, -15,  -4,   1,   1,  -4, -15, -31, 0, 0, 0, 0, 0, 0, 0, 0,
 -53, -37, -26, -22, -22, -26, -37, -53, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // bishop
 -1, -3, -6, -8, -8, -6, -3, -1, 0, 0, 0, 0, 0, 0, 0, 0,
 -3,  4,  1, -1, -1,  1,  4, -3, 0, 0, 0, 0, 0, 0, 0, 0,
 -6,  1,  8,  7,  7,  8,  1, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -8, -1,  7, 16, 16,  7, -1, -8, 0, 0, 0, 0, 0, 0, 0, 0,
 -8, -1,  7, 16, 16,  7, -1, -8, 0, 0, 0, 0, 0, 0, 0, 0,
 -6,  1,  8,  7,  7,  8,  1, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -3,  4,  1, -1, -1,  1,  4, -3, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -8,-11,-13,-13,-11, -8, -6, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // rook
 -7, -3,  1,  5,  5,  1, -3, -7, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -2,  2,  6,  6,  2, -2, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -2,  2,  6,  6,  2, -2, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -2,  2,  6,  6,  2, -2, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -2,  2,  6,  6,  2, -2, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -2,  2,  6,  6,  2, -2, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -2,  2,  6,  6,  2, -2, -6, 0, 0, 0, 0, 0, 0, 0, 0,
 -6, -2,  2,  6,  6,  2, -2, -6, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // queen
-11, -7, -4, -2, -2, -4, -7,-11, 0, 0, 0, 0, 0, 0, 0, 0,
 -7, -1,  2,  4,  4,  2, -1, -7, 0, 0, 0, 0, 0, 0, 0, 0,
 -4,  2,  6,  7,  7,  6,  2, -4, 0, 0, 0, 0, 0, 0, 0, 0,
 -2,  4,  7, 10, 10,  7,  4, -2, 0, 0, 0, 0, 0, 0, 0, 0,
 -2,  4,  7, 10, 10,  7,  4, -2, 0, 0, 0, 0, 0, 0, 0, 0,
 -4,  2,  6,  7,  7,  6,  2, -4, 0, 0, 0, 0, 0, 0, 0, 0,
 -7, -1,  2,  4,  4,  2, -1, -7, 0, 0, 0, 0, 0, 0, 0, 0,
-16,-12, -9, -7, -7, -9,-12,-16, 0, 0, 0, 0, 0, 0, 0, 0, },

{ // king
 -8, -3,-33,-52,-52,-33, -3, -8, 0, 0, 0, 0, 0, 0, 0, 0,
  2,  8,-22,-42,-42,-22,  8,  2, 0, 0, 0, 0, 0, 0, 0, 0,
 12, 18,-12,-32,-32,-12, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0,
 17, 23, -7,-27,-27, -7, 23, 17, 0, 0, 0, 0, 0, 0, 0, 0,
 22, 28, -2,-22,-22, -2, 28, 22, 0, 0, 0, 0, 0, 0, 0, 0,
 25, 31,  1,-19,-19,  1, 31, 25, 0, 0, 0, 0, 0, 0, 0, 0,
 28, 33,  4,-16,-16,  4, 33, 28, 0, 0, 0, 0, 0, 0, 0, 0,
 31, 36,  6,-14,-14,  6, 36, 31, 0, 0, 0, 0, 0, 0, 0, 0, },
// mirror piece tables are filled in during init_eval
};
void remove_piece(position_t* pos, square_t square)
{
    assert(pos->board[square]);
    piece_t piece = pos->board[square];
    color_t color = piece_color(piece);
    
    if (piece_is_type(piece, PAWN)) {
        int index = --pos->num_pawns[color];
        int position = pos->piece_index[square];
        square_t sq = pos->pawns[color][index];
        pos->pawns[color][position] = sq;
        pos->pawns[color][index] = INVALID_SQUARE;
        pos->piece_index[sq] = position;
        pos->pawn_hash ^= piece_hash(piece, square);
    } else {
        int index = --pos->num_pieces[color];
        for (; index>0 && pos->pieces[color][index] != square; --index) {}
        for (; index <= pos->num_pieces[color]; ++index) {
            square_t sq = pos->pieces[color][index+1];
            pos->pieces[color][index] = sq;
            pos->piece_index[sq] = index;
        }
    }
    pos->board[square] = EMPTY;
    pos->piece_index[square] = -1;
    pos->piece_count[piece]--;
    pos->hash ^= piece_hash(piece, square);
    pos->material_hash ^= material_hash(piece, pos->piece_count[piece]);
#define material_value(piece)               material_values[piece]
    pos->material_eval[color] -= material_value(piece);
#define piece_square_value(piece, square)   piece_square_values[piece][square]
    pos->piece_square_eval[color].midgame -= piece_square_value(piece, square);
#define endgame_piece_square_value(piece, square) \
    endgame_piece_square_values[piece][square]
    pos->piece_square_eval[color].endgame -=
        endgame_piece_square_value(piece, square);
}
void transfer_piece(position_t* pos, square_t from, square_t to)
{
    assert(pos->board[from]);
    if (from == to) return;
    if (pos->board[to] != EMPTY) {
        remove_piece(pos, to);
    }

    piece_t p = pos->board[from];
    pos->board[to] = pos->board[from];
    int index = pos->piece_index[to] = pos->piece_index[from];
    color_t color = piece_color(p);
    pos->board[from] = EMPTY;
    if (piece_is_type(p, PAWN)) {
        pos->pawns[color][index] = to;
        pos->piece_index[to] = index;
        pos->pawn_hash ^= piece_hash(p, from);
        pos->pawn_hash ^= piece_hash(p, to);
    } else {
        pos->pieces[color][index] = to;
        pos->piece_index[to] = index;
    }
    pos->piece_square_eval[color].midgame -= piece_square_value(p, from);
    pos->piece_square_eval[color].midgame += piece_square_value(p, to);
    pos->piece_square_eval[color].endgame -=
        endgame_piece_square_value(p, from);
    pos->piece_square_eval[color].endgame +=
        endgame_piece_square_value(p, to);
    pos->hash ^= piece_hash(p, from);
    pos->hash ^= piece_hash(p, to);
}
typedef enum {
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NONE
} file_tag_t;
typedef int file_t;
/*
 * Modify |pos| by adding |piece| to |square|. If |square| is occupied, its
 * occupant is properly removed.
 */
void place_piece(position_t* pos, piece_t piece, square_t square)
{
    if (pos->board[square]) {
        remove_piece(pos, square);
    }
    color_t color = piece_color(piece);
    piece_type_t type = piece_type(piece);
    (void)type;
    assert(color == WHITE || color == BLACK);
    assert(type >= PAWN && type <= KING);
    assert(square != INVALID_SQUARE);

    pos->board[square] = piece;
    if (piece_is_type(piece, PAWN)) {
        int index = pos->num_pawns[color]++;
        pos->pawns[color][index] = square;
        pos->piece_index[square] = index;
        pos->pawn_hash ^= piece_hash(piece, square);
    } else {
        int index = pos->num_pieces[color]++;
        pos->pieces[color][index+1] = INVALID_SQUARE;
        for (; index>0 && pos->board[pos->pieces[color][index-1]] < piece;
                --index) {
            square_t sq = pos->pieces[color][index-1];
            pos->pieces[color][index] = sq;
            pos->piece_index[sq] = index;
        }
        pos->pieces[color][index] = square;
        pos->piece_index[square] = index;
    }
    pos->hash ^= piece_hash(piece, square);
    pos->material_hash ^= material_hash(piece, pos->piece_count[piece]);
    pos->material_eval[color] += material_value(piece);
    pos->piece_square_eval[color].midgame += piece_square_value(piece, square);
    pos->piece_square_eval[color].endgame +=
        endgame_piece_square_value(piece, square);
    pos->piece_count[piece]++;
}
/*
 * Set |pos->check_square| to the location of a checking piece, and return 0
 * if |pos| is not check, 1 if there is exactly 1 checker, or 2 if there are
 * multiple checkers.
 */
uint8_t find_checks(position_t* pos)
{
    // For every opposing piece, look up the attack data for its square.
    uint8_t attackers = 0;
    pos->check_square = EMPTY;
    color_t side = pos->side_to_move^1;
    square_t sq = pos->pieces[side^1][0];

    // Special-case pawns for efficiency.
    piece_t opp_pawn = create_piece(side, PAWN);
    if (pos->board[sq - piece_deltas[opp_pawn][0]] == opp_pawn) {
        pos->check_square = sq - piece_deltas[opp_pawn][0];
        if (++attackers > 1) return attackers;
    }
    if (pos->board[sq - piece_deltas[opp_pawn][1]] == opp_pawn) {
        pos->check_square = sq - piece_deltas[opp_pawn][1];
        if (++attackers > 1) return attackers;
    }

    square_t from;
    for (square_t* pfrom = &pos->pieces[side][1];
            (from = *pfrom) != INVALID_SQUARE;
            ++pfrom) {
        piece_t p = pos->board[from];
        if (possible_attack(from, sq, p)) {
            if (piece_slide_type(p) == NO_SLIDE) {
                pos->check_square = from;
                if (++attackers > 1) return attackers;
                continue;
            }
            square_t att_sq = from;
            direction_t att_dir = direction(from, sq);
            while (att_sq != sq) {
                att_sq += att_dir;
                if (att_sq == sq) {
                    pos->check_square = from;
                    if (++attackers > 1) return attackers;
                }
                if (pos->board[att_sq]) break;
            }
        }
    }
    return attackers;
}
void do_move(position_t* pos, move_t move, undo_info_t* undo)
{
#define check_move_validity(x,y)                ((void)0,(void)0)
    check_move_validity(pos, move);
    check_board_validity(pos);
    // Set undo info, so we can roll back later.
    undo->is_check = pos->is_check;
    undo->check_square = pos->check_square;
    undo->prev_move = pos->prev_move;
    undo->ep_square = pos->ep_square;
    undo->fifty_move_counter = pos->fifty_move_counter;
    undo->castle_rights = pos->castle_rights;
    undo->hash = pos->hash;

    // xor old data out of the hash key
    pos->hash ^= ep_hash(pos);
    pos->hash ^= castle_hash(pos);
    pos->hash ^= side_hash(pos);

    const color_t side = pos->side_to_move;
    const color_t other_side = side^1;
    const square_t from = get_move_from(move);
    const square_t to = get_move_to(move);
    assert(valid_board_index(from) && valid_board_index(to));
    pos->ep_square = EMPTY;
    ++pos->fifty_move_counter;
    if (piece_type(get_move_piece(move)) == PAWN) {
#define square_rank(square)         ((square) >> 4)
#define square_file(square)         ((square) & 0x0f)
#define square_color(square)        (((square) ^ ((square) >> 4) ^ 1) & 1)
#define create_square(file, rank)   (((rank) << 4) | (file))
#define valid_board_index(idx)      !((idx) & 0x88)
#define flip_square(square)         ((square) ^ 0x70)
#define mirror_rank(square)         ((square) ^ 0x70)
#define mirror_file(square)         ((square) ^ 0x07)
#define square_to_index(square)     ((square)+((square) & 0x07))>>1
#define index_to_square(index)      ((index)+((index) & ~0x07))

        if (relative_rank[side][square_rank(to)] -
                relative_rank[side][square_rank(from)] != 1) {
            piece_t opp_pawn = create_piece(other_side, PAWN);
            if (pos->board[to-1] == opp_pawn || pos->board[to+1] == opp_pawn) {
                pos->ep_square = from + pawn_push[side];
            }
        }
        pos->fifty_move_counter = 0;
    } else if (get_move_capture(move) != EMPTY) {
        pos->fifty_move_counter = 0;
    }

    // Remove castling rights as necessary.
    if (from == (square_t)(queen_rook_home + side*A8)) {
        remove_ooo_rights(pos, side);
    } else if (from == (square_t)(king_rook_home + side*A8)) {
        remove_oo_rights(pos, side);
    } else if (from == (square_t)(king_home + side*A8))  {
        remove_oo_rights(pos, side);
        remove_ooo_rights(pos, side);
    } 
    if (to == (square_t)(queen_rook_home + other_side*A8)) {
        remove_ooo_rights(pos, other_side);
    } else if (to == (square_t)(king_rook_home + other_side*A8)) {
        remove_oo_rights(pos, other_side);
    }

    if (!is_move_castle(move)) transfer_piece(pos, from, to);

    const piece_type_t promote_type = get_move_promote(move);
#define is_move_castle_short(move) \
    (is_move_castle(move) && (square_file(get_move_to(move)) == FILE_G))
    if (is_move_castle_short(move)) {
        assert(pos->board[king_home + A8*side] == create_piece(side, KING));
        assert(pos->board[king_rook_home + A8*side] ==
                create_piece(side, ROOK));
        remove_piece(pos, king_home + A8*side);
        transfer_piece(pos, king_rook_home + A8*side, F1 + A8*side);
        place_piece(pos, create_piece(side, KING), G1 + A8*side);
    } else if (is_move_castle_long(move)) {
        assert(pos->board[king_home + A8*side] == create_piece(side, KING));
        assert(pos->board[queen_rook_home + A8*side] ==
                create_piece(side, ROOK));
        remove_piece(pos, king_home + A8*side);
        transfer_piece(pos, queen_rook_home + A8*side, D1 + A8*side);
        place_piece(pos, create_piece(side, KING), C1 + A8*side);
    } else if (is_move_enpassant(move)) {
        remove_piece(pos, to-pawn_push[side]);
    } else if (promote_type) {
        place_piece(pos, create_piece(side, promote_type), to);
    }

    pos->hash_history[pos->ply++] = undo->hash;
    assert(pos->ply <= HASH_HISTORY_LENGTH);
    pos->side_to_move ^= 1;
    pos->is_check = find_checks(pos);
    pos->prev_move = move;
    pos->hash ^= ep_hash(pos);
    pos->hash ^= castle_hash(pos);
    pos->hash ^= side_hash(pos);
    check_board_validity(pos);
}
/*
 * Undo the effects of |move| on |pos|, using the undo info generated by
 * do_move().
 */
void undo_move(position_t* pos, move_t move, undo_info_t* undo)
{
    check_board_validity(pos);
    const color_t side = pos->side_to_move^1;
    const square_t from = get_move_from(move);
    const square_t to = get_move_to(move);

    // Move the piece back, and fix en passant captures.
    if (!is_move_castle(move)) transfer_piece(pos, to, from);
    piece_type_t captured = get_move_capture(move);
    if (captured != EMPTY) {
        if (is_move_enpassant(move)) {
            place_piece(pos, create_piece(side^1, PAWN), to-pawn_push[side]);
        } else {
            place_piece(pos, create_piece(pos->side_to_move, captured), to);
        }
    }

    // Un-promote/castle, if necessary.
    const piece_type_t promote_type = get_move_promote(move);
    if (is_move_castle_short(move)) {
        remove_piece(pos, G1 + A8*side);
        transfer_piece(pos, F1 + A8*side, king_rook_home + A8*side);
        place_piece(pos, create_piece(side, KING), king_home + A8*side);
    } else if (is_move_castle_long(move)) {
        remove_piece(pos, C1 + A8*side);
        transfer_piece(pos, D1 + A8*side, queen_rook_home + A8*side);
        place_piece(pos, create_piece(side, KING), king_home + A8*side);
    } else if (promote_type) {
        place_piece(pos, create_piece(side, PAWN), from);
    }

    // Reset non-board state information.
    pos->side_to_move ^= 1;
    pos->ply--;
    pos->is_check = undo->is_check;
    pos->check_square = undo->check_square;
    pos->ep_square = undo->ep_square;
    pos->fifty_move_counter = undo->fifty_move_counter;
    pos->castle_rights = undo->castle_rights;
    pos->prev_move = undo->prev_move;
    pos->hash = undo->hash;
    check_board_validity(pos);
}
/*
 * Test a pseudo-legal move's legality. For this, we only have to check that
 * it doesn't leave the king in check.
 */
bool is_pseudo_move_legal(position_t* pos, move_t move)
{
    if (!move) return false;
    piece_t piece = get_move_piece(move);
    square_t to = get_move_to(move);
    square_t from = get_move_from(move);
    color_t side = piece_color(piece);
    if (piece_is_type(piece, KING)) return !is_square_attacked(pos, to, side^1);

    // Avoid moving pinned pieces.
    direction_t pin_dir = pin_direction(pos, from, pos->pieces[side][0]);
    if (pin_dir) return abs(pin_dir) == abs(direction(from, to));

    // Resolving pins for en passant moves is a pain, and they're very rare.
    // I just do them the expensive way and don't worry about it.
    if (is_move_enpassant(move)) {
        undo_info_t undo;
        do_move(pos, move, &undo);
        bool legal = !is_square_attacked(pos, pos->pieces[side][0], side^1);
        undo_move(pos, move, &undo);
        return legal;
    }
    return true;
}

/*
 * Push a new move onto a stack of moves, first doing some sanity checks.
 */
static move_t* add_move(const position_t* pos,
        move_t move,
        move_t* moves)
{
    (void)pos; // avoid warning when NDEBUG is defined
    check_move_validity(pos, move);
    *(moves++) = move;
    return moves;
}

/*
 * Generate all moves that evade check in the given position. This is purely
 * legal move generation; no pseudo-legal moves.
 */
int generate_evasions(const position_t* pos, move_t* moves)
{
    assert(pos->is_check && pos->board[pos->check_square]);
    move_t* moves_head = moves;
    color_t side = pos->side_to_move, other_side = side^1;
    square_t king_sq = pos->pieces[side][0];
    square_t check_sq = pos->check_square;
    piece_t king = create_piece(side, KING);
    piece_t checker = pos->board[check_sq];

    // Generate king moves.
    // Don't let the king mask its possible destination squares in calls
    // to is_square_attacked.
    square_t from = king_sq, to = INVALID_SQUARE;
    ((position_t*)pos)->board[king_sq] = EMPTY;
    for (const direction_t* delta = piece_deltas[king]; *delta; ++delta) {
        to = from + *delta;
        piece_t capture = pos->board[to];
        if (capture != EMPTY && !can_capture(king, capture)) continue;
        if (is_square_attacked((position_t*)pos,to,other_side)) continue;
        ((position_t*)pos)->board[king_sq] = king;
        moves = add_move(pos, create_move(from, to, king, capture), moves);
        ((position_t*)pos)->board[king_sq] = EMPTY;
    }
    ((position_t*)pos)->board[king_sq] = king;
    // If there are multiple checkers, only king moves are possible.
    if (pos->is_check > 1) {
        *moves = 0;
        return moves-moves_head;
    }
    
    // First, the most common case: a check that can be evaded via an
    // en passant capture. Note that if we're in check and an en passant
    // capture is available, the only way the en passant capture would evade
    // the check is if it's the newly moved pawn that's checking us.
    direction_t pin_dir;
    if (pos->ep_square != EMPTY &&
            check_sq+pawn_push[side] == pos->ep_square &&
            pos->board[pos->ep_square] == EMPTY) {
        piece_t our_pawn = create_piece(side, PAWN);
        to = pos->ep_square;
        for (int i=0; i<2; ++i) {
            from = to-piece_deltas[our_pawn][i];
            if (pos->board[from] && pos->board[from] == our_pawn) {
                pin_dir = pin_direction(pos, from, king_sq);
                if (pin_dir) continue;
                moves = add_move(pos,
                        create_move_enpassant(from, to, our_pawn, checker),
                        moves);
            }
        }
    }
    // Generate captures of the checker.
    for (int i = 0; i < pos->num_pawns[side]; ++i) {
        from = pos->pawns[side][i];
        piece_t piece = create_piece(side, PAWN);
        pin_dir = pin_direction(pos, from, king_sq);
        if (pin_dir) continue;
        if (!possible_attack(from, check_sq, piece)) continue;
        if (relative_rank[side][square_rank(from)] == RANK_7) {
            // Capture and promote.
            for (piece_t promoted=QUEEN; promoted > PAWN; --promoted) {
                moves = add_move(pos,
                        create_move_promote(from, check_sq, piece,
                            checker, promoted),
                        moves);
            }
        } else {
            moves = add_move(pos,
                    create_move(from, check_sq, piece, checker),
                    moves);
        }
    }
    for (int i = 1; i < pos->num_pieces[side]; ++i) {
        from = pos->pieces[side][i];
        piece_t piece = pos->board[from];
        pin_dir = pin_direction(pos, from, king_sq);
        if (pin_dir) continue;
        if (!possible_attack(from, check_sq, piece)) continue;
        if (piece_slide_type(piece) == NONE) {
            moves = add_move(pos,
                    create_move(from, check_sq, piece, checker),
                    moves);
        } else {
            // A sliding piece, keep going until we hit something.
            direction_t check_dir = direction(from, check_sq);
            for (to=from+check_dir; pos->board[to] == EMPTY; to+=check_dir) {}
            if (to == check_sq) {
                moves = add_move(pos,
                        create_move(from, to, piece, checker),
                        moves);
                continue;
            }
        }
    }
    if (piece_slide_type(checker) == NONE) {
        *moves = 0;
        return moves-moves_head;
    }
    
    // A slider is doing the checking; generate blocking moves.
    direction_t block_dir = direction(check_sq, king_sq);
    for (int i = 0; i < pos->num_pawns[side]; ++i) {
        from = pos->pawns[side][i];
        piece_t piece = create_piece(side, PAWN);
        direction_t k_dir;
        pin_dir = pin_direction(pos, from, king_sq);
        if (pin_dir) continue;
        to = from + pawn_push[side];
        if (pos->board[to] != EMPTY) continue;
        rank_t rank = relative_rank[side][square_rank(from)];
        k_dir = direction(to, king_sq);
        if (k_dir == block_dir &&
                ((king_sq < to && check_sq > to) ||
                 (king_sq > to && check_sq < to))) {
            if (rank == RANK_7) {
                // Block and promote.
                for (piece_t promoted=QUEEN; promoted > PAWN; --promoted) {
                    moves = add_move(pos,
                        create_move_promote(from, to, piece, EMPTY, promoted),
                        moves);
                }
            } else {
                moves = add_move(pos,
                        create_move(from, to, piece, EMPTY),
                        moves);
            }
        }
        if (rank != RANK_2) continue;
        to += pawn_push[side];
        if (pos->board[to]) continue;
        k_dir = direction(to, king_sq);
        if (k_dir == block_dir &&
                ((king_sq < to && check_sq > to) ||
                 (king_sq > to && check_sq < to))) {
            moves = add_move(pos,
                    create_move(from, to, piece, EMPTY),
                    moves);
        }
    }

    for (int i=1; i<pos->num_pieces[side]; ++i) {
        from = pos->pieces[side][i];
        piece_t piece = pos->board[from];
        direction_t k_dir;
        pin_dir = pin_direction(pos, from, king_sq);
        if (pin_dir) continue;
        if (piece_is_type(piece, KNIGHT)) {
            for (const direction_t* delta=piece_deltas[piece];
                    *delta; ++delta) {
                to = from + *delta;
                if (pos->board[to] != EMPTY) continue;
                k_dir = direction(to, king_sq);
                if (k_dir == block_dir &&
                        ((king_sq < to && check_sq > to) ||
                         (king_sq > to && check_sq < to))) {
                    moves = add_move(pos,
                            create_move(from, to, piece, EMPTY),
                            moves);
                }
            }
        } else {
           for (const direction_t* delta=piece_deltas[piece];
                   *delta; ++delta) {
                for (to = from+*delta; pos->board[to] == EMPTY; to+=*delta) {
                    k_dir = direction(to, king_sq);
                    if (k_dir == block_dir &&
                            ((king_sq < to && check_sq > to) ||
                             (king_sq > to && check_sq < to))) {
                        moves = add_move(pos,
                                create_move(from, to, piece, EMPTY),
                                moves);
                        break;
                    }
                }
            }
        }
    }
    *moves = 0;
    return moves-moves_head;
}

#define material_value(piece)               material_values[piece]
#define eg_material_value(piece)            eg_material_values[piece]
#define piece_square_value(piece, square)   piece_square_values[piece][square]
#define endgame_piece_square_value(piece, square) \
    endgame_piece_square_values[piece][square]
/*
 * In |pos|, is the side to move in check?
 */
bool is_check(const position_t* pos)
{
    return pos->is_check;
}
/*
 * Add all pseudo-legal captures that the given pawn can make.
 */
static void generate_pawn_captures(const position_t* pos,
        square_t from,
        piece_t piece,
        move_t** moves_head)
{
    color_t side = pos->side_to_move;
    int cap_left = side == WHITE ? 15 : -15;
    int cap_right = side == WHITE ? 17 : -17;
    square_t to;
    rank_t r_rank = relative_rank[side][square_rank(from)];
    move_t* moves = *moves_head;
    if (r_rank < RANK_7) {
        // non-promote captures
        to = from + cap_left;
        if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
            moves = add_move(pos, create_move(from, to, piece,
                    pos->board[to]), moves);
        } else if (to == pos->ep_square && pos->board[to] == EMPTY) {
            moves = add_move(pos, create_move_enpassant(from, to, piece,
                    pos->board[to + pawn_push[side^1]]), moves);
        }
        to = from + cap_right;
        if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
            moves = add_move(pos, create_move(from, to, piece,
                    pos->board[to]), moves);
        } else if (to == pos->ep_square && pos->board[to] == EMPTY) {
            moves = add_move(pos, create_move_enpassant(from, to, piece,
                    pos->board[to + pawn_push[side^1]]), moves);
        }
    } else {
        // capture/promotes
        to = from + cap_left;
        if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
            for (piece_t promoted=QUEEN; promoted > PAWN; --promoted) {
                moves = add_move(pos, create_move_promote(from, to, piece,
                        pos->board[to], promoted), moves);
            }
        }
        to = from + cap_right;
        if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
            for (piece_t promoted=QUEEN; promoted > PAWN; --promoted) {
                moves = add_move(pos, create_move_promote(from, to, piece,
                        pos->board[to], promoted), moves);
            }
        }
    }
    *moves_head = moves;
}
/*
 * Add all pseudo-legal captures that the given piece (N/B/Q/K) can make.
 */
static void generate_piece_captures(const position_t* pos,
        square_t from,
        piece_t piece,
        move_t** moves_head)
{
    move_t* moves = *moves_head;
    square_t to;
    // Note: I unrolled all these loops to handle each direction explicitly.
    // The idea was to increase performance, but it's only about 1% faster
    // for much more code, so it's possible I'll change this back later.
    switch (piece_type(piece)) {
        case KING:
            to = from - 17;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from - 16;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from - 15;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from - 1;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 1;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 15;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 16;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 17;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            break;
        case KNIGHT:
            to = from - 33;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from - 31;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from - 18;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from - 14;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 14;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 18;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 31;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            to = from + 33;
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            break;
        case BISHOP:
            for (to=from-17; pos->board[to]==EMPTY; to-=17) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from-15; pos->board[to]==EMPTY; to-=15) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+15; pos->board[to]==EMPTY; to+=15) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+17; pos->board[to]==EMPTY; to+=17) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            break;
        case ROOK:
            for (to=from-16; pos->board[to]==EMPTY; to-=16) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from-1; pos->board[to]==EMPTY; to-=1) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+1; pos->board[to]==EMPTY; to+=1) {}
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+16; pos->board[to]==EMPTY; to+=16) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            break;
        case QUEEN:
            for (to=from-17; pos->board[to]==EMPTY; to-=17) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from-16; pos->board[to]==EMPTY; to-=16) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from-15; pos->board[to]==EMPTY; to-=15) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from-1; pos->board[to]==EMPTY; to-=1) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+1; pos->board[to]==EMPTY; to+=1) {}
            if (pos->board[to] != EMPTY && can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+15; pos->board[to]==EMPTY; to+=15) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+16; pos->board[to]==EMPTY; to+=16) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            for (to=from+17; pos->board[to]==EMPTY; to+=17) {}
            if (can_capture(piece, pos->board[to])) {
                moves = add_move(pos, create_move(from, to, piece,
                            pos->board[to]), moves);
            }
            break;
        default: assert(false);
    }
    *moves_head = moves;
}
/*
 * Fill the provided list with all pseudolegal captures in the given position.
 */
static int generate_pseudo_captures(const position_t* pos, move_t* moves)
{
    move_t* moves_head = moves;
    color_t side = pos->side_to_move;
    piece_t piece;
    square_t from;
    for (int i = 0; i < pos->num_pieces[side]; ++i) {
        from = pos->pieces[side][i];
        piece = pos->board[from];
        generate_piece_captures(pos, from, piece, &moves);
    }
    piece = create_piece(side, PAWN);
    for (int i=0; i<pos->num_pawns[side]; ++i) {
        from = pos->pawns[side][i];
        generate_pawn_captures(pos, from, piece, &moves);
    }
    *moves = 0;
    return moves-moves_head;
}
/*
 * Add all pseudo-legal non-capturing promotions.
 */
static int generate_promotions(const position_t* pos, move_t* moves)
{
    move_t* moves_head = moves;
    color_t side = pos->side_to_move;
    piece_t piece = create_piece(side, PAWN);
    for (int i = 0; i < pos->num_pawns[side]; ++i) {
        square_t from = pos->pawns[side][i];
        rank_t r_rank = relative_rank[side][square_rank(from)];
        if (r_rank < RANK_7) continue;
        square_t to = from + pawn_push[side];
        if (pos->board[to]) continue;
        for (piece_type_t type=KNIGHT; type<=QUEEN; ++type) {
            moves = add_move(pos,
                    create_move_promote(from, to, piece, EMPTY, type),
                    moves);
        }
    }
    *moves = 0;
    return (moves-moves_head);
}
/*
 * Generate pseudo-legal captures and promotions. These moves are considered
 * first by the move selection algorithm.
 */
int generate_pseudo_tactical_moves(const position_t* pos, move_t* moves)
{
    move_t* moves_head = moves;
    moves += generate_promotions(pos, moves);
    moves += generate_pseudo_captures(pos, moves);
    return moves - moves_head;
}
/*
 * Add all pseudo-legal non-capturing moves that the given piece (N/B/Q/K)
 * can make, with the exception of castles.
 */
static void generate_piece_noncaptures(const position_t* pos,
        square_t from,
        piece_t piece,
        move_t** moves_head)
    {
        move_t* moves = *moves_head;
        square_t to;
        if (piece_slide_type(piece) == NONE) {
            // not a sliding piece, just iterate over dest. squares
            for (const direction_t* delta = piece_deltas[piece]; *delta; ++delta) {
                to = from + *delta;
                if (pos->board[to] != EMPTY) continue;
                moves = add_move(pos,
                        create_move(from, to, piece, NONE),
                        moves);
            }
        } else {
            // a sliding piece, keep going until we hit something
            for (const direction_t* delta = piece_deltas[piece]; *delta; ++delta) {
                to = from + *delta;
                while (pos->board[to] == EMPTY) {
                    moves = add_move(pos,
                            create_move(from, to, piece, NONE),
                            moves);
                    to += *delta;
                }
            }
        }
        *moves_head = moves;
    }
/*
 * Generate all non-capturing, non-promoting pawn moves.
 */
static void generate_pawn_quiet_moves(const position_t* pos,
        square_t from,
        piece_t piece,
        move_t** moves_head)
{
    color_t side = pos->side_to_move;
    square_t to;
    rank_t r_rank = relative_rank[side][square_rank(from)];
    move_t* moves = *moves_head;
    to = from + pawn_push[side];
    if (r_rank == RANK_7 || pos->board[to] != EMPTY) return;
    moves = add_move(pos, create_move(from, to, piece, EMPTY), moves);
    to += pawn_push[side];
    if (r_rank == RANK_2 && pos->board[to] == EMPTY) {
        // initial two-square push
        moves = add_move(pos,
                create_move(from, to, piece, EMPTY),
                moves);
    }
    *moves_head = moves;
}
/*
 * Generate pseudo-legal moves which are neither captures nor promotions.
 */
int generate_pseudo_quiet_moves(const position_t* pos, move_t* moves)
{
    move_t* moves_head = moves;
    color_t side = pos->side_to_move;
    piece_t piece;
    square_t from;
#define	MIN(a,b) (((a)<(b))?(a):(b))
#define	MAX(a,b) (((a)>(b))?(a):(b))

    // Castling. Castles are considered pseudo-legal if we have appropriate
    // castling rights, the squares between king and rook are unoccupied,
    // and the intermediate square is unattacked. Therefore checking for
    // legality just requires seeing if we're in check afterwards.
    // This is messy for Chess960, so it's separated into separate cases.
    square_t my_king_home = king_home + side*A8;
        if (has_oo_rights(pos, side)) {
            square_t my_f1 = F1 + side*A8;
            square_t my_g1 = G1 + side*A8;
            square_t my_kr = king_rook_home + side*A8;
            bool castle_ok = true;
            // Check that rook is unimpeded.
            for (square_t sq = MIN(my_kr, my_f1);
                    sq <= MAX(my_kr, my_f1); ++sq) {
                if (sq != my_kr  && sq != my_king_home &&
                        pos->board[sq] != EMPTY) {
                    castle_ok = false;
                    break;
                }
            }
            // Check that the king is unimpeded and unattacked
            if (castle_ok) {
                for (square_t sq = MIN(my_king_home, my_g1);
                        sq <= my_g1; ++sq) {
                    if (sq != my_king_home && sq != my_kr &&
                            pos->board[sq] != EMPTY) {
                        castle_ok = false;
                        break;
                    }
                    if (sq != my_g1 &&
                            is_square_attacked((position_t*)pos, sq, side^1)) {
                        castle_ok = false;
                        break;
                    }
                }
            }
            if (castle_ok) moves = add_move(pos,
                    create_move_castle(my_king_home, my_g1,
                        create_piece(side, KING)), moves);
        }
        if (has_ooo_rights(pos, side)) {
            square_t my_d1 = D1 + side*A8;
            square_t my_c1 = C1 + side*A8;
            square_t my_qr = queen_rook_home + side*A8;
            bool castle_ok = true;
            // Check that rook is unimpeded.
            for (square_t sq = MIN(my_qr, my_d1);
                    sq <= MAX(my_qr, my_d1); ++sq) {
                if (sq != my_qr && sq != my_king_home &&
                        pos->board[sq] != EMPTY) {
                    castle_ok = false;
                    break;
                }
            }
            // Check that the king is unimpeded and unattacked
            if (castle_ok) {
                for (square_t sq = MIN(my_king_home, my_c1);
                        sq <= MAX(my_king_home, my_c1); ++sq) {
                    if (sq != my_king_home && sq != my_qr &&
                            pos->board[sq] != EMPTY) {
                        castle_ok = false;
                        break;
                    }
                    if (sq != my_c1 &&
                            is_square_attacked((position_t*)pos, sq, side^1)) {
                        castle_ok = false;
                        break;
                    }
                }
            }
            if (castle_ok) moves = add_move(pos,
                    create_move_castle(my_king_home, my_c1,
                        create_piece(side, KING)),
                    moves);
        }

    for (int i = 0; i < pos->num_pieces[side]; ++i) {
        from = pos->pieces[side][i];
        piece = pos->board[from];
        assert(piece_color(piece) == side && piece_type(piece) != PAWN);
        generate_piece_noncaptures(pos, from, piece, &moves);
    }
    piece = create_piece(side, PAWN);
    for (int i=0; i<pos->num_pawns[side]; ++i) {
        from = pos->pawns[side][i];
        assert(pos->board[from] == piece);
        generate_pawn_quiet_moves(pos, from, piece, &moves);
    }

    *moves = 0;
    return (moves-moves_head);
}
/*
 * Fill the provided list with all pseudolegal moves in the given position.
 * Pseudolegal moves are moves which would be legal if we didn't have to worry
 * about leaving our king in check.
 */
int generate_pseudo_moves(const position_t* pos, move_t* moves)
{
    if (is_check(pos)) return generate_evasions(pos, moves);
    move_t* moves_head = moves;
    moves += generate_pseudo_tactical_moves(pos, moves);
    moves += generate_pseudo_quiet_moves(pos, moves);
    return moves-moves_head;
}
/*
 * Fill the provided list with all legal moves in the given position.
 */
int generate_legal_moves(position_t* pos, move_t* moves)
{
    if (is_check(pos)) return generate_evasions(pos, moves);
    int num_pseudo = generate_pseudo_moves(pos, moves);
    move_t* moves_tail = moves+num_pseudo;
    move_t* moves_curr = moves;
    while (moves_curr < moves_tail) {
        check_pseudo_move_legality(pos, *moves_curr);
        if (!is_pseudo_move_legal(pos, *moves_curr)) {
            *moves_curr = *(--moves_tail);
            *moves_tail = 0;
        } else {
            ++moves_curr;
        }
    }
    return moves_tail-moves;
}

typedef uint64_t bitboard_t;
typedef enum {
    EG_NONE,
    EG_WIN,
    EG_DRAW,
    EG_KBNK,
    EG_KPK,
    EG_LAST
} endgame_type_t;
typedef struct {
    bitboard_t pawns_bb[2];
    bitboard_t outposts_bb[2];
    bitboard_t passed_bb[2];
    square_t passed[2][8];
    int num_passed[2];
    score_t score[2];
    int kingside_storm[2];
    int queenside_storm[2];
    hashkey_t key;
} pawn_data_t;
typedef struct {
    hashkey_t key;
    endgame_type_t eg_type;
    int phase;
    int scale[2];
    score_t score;
    color_t strong_side;
} material_data_t;

typedef struct {
    pawn_data_t* pd;
    material_data_t* md;
} eval_data_t;
static material_data_t* material_table = NULL;
static int num_buckets;
static struct {
    int misses;
    int hits;
    int occupied;
    int evictions;
} material_hash_stats;
/*
 * Is this position an opening or an endgame? Scored on a scale of 0-24,
 * with 24 being a pure opening and 0 a pure endgame.
 * Note: the maximum phase is given by MAX_PHASE, which needs to be updated
 *       in conjunction with this function.
 */
int game_phase(const position_t* pos)
{
    return pos->piece_count[WN] + pos->piece_count[WB] +
        2*pos->piece_count[WR] + 4*pos->piece_count[WQ] +
        pos->piece_count[BN] + pos->piece_count[BB] +
        2*pos->piece_count[BR] + 4*pos->piece_count[BQ];
}
/*
 * Calculate static score adjustments and scaling factors that are based
 * solely on the combination of pieces on the board. Each combination
 * has an associated hash key so that this data can be cached in in the
 * material table. There are relatively few possible material configurations
 * reachable in a typical search, so the hit rate should be extremely high.
 */
static void compute_material_data(const position_t* pos, material_data_t* md)
{
    md->phase = game_phase(pos);

    md->score.midgame = 0;
    md->score.endgame = 0;

    int wp = pos->piece_count[WP];
    int bp = pos->piece_count[BP];
    int wn = pos->piece_count[WN];
    int bn = pos->piece_count[BN];
    int wb = pos->piece_count[WB];
    int bb = pos->piece_count[BB];
    int wr = pos->piece_count[WR];
    int br = pos->piece_count[BR];
    int wq = pos->piece_count[WQ];
    int bq = pos->piece_count[BQ];
    int w_major = 2*wq + wr;
    int w_minor = wn + wb;
    int w_piece = 2*w_major + w_minor;
    int w_all = wq + wr + wb + wn + wp;
    int b_major = 2*bq + br;
    int b_minor = bn + bb;
    int b_piece = 2*b_major + b_minor;
    int b_all = bq + br + bb + bn + bp;

    // Pair bonuses
    if (wb > 1) {
        md->score.midgame += 30;
        md->score.endgame += 45;
    }
    if (bb > 1) {
        md->score.midgame -= 30;
        md->score.endgame -= 45;
    }
    if (wr > 1) {
        md->score.midgame -= 12;
        md->score.endgame -= 17;
    }
    if (br > 1) {
        md->score.midgame += 12;
        md->score.endgame += 17;
    }
    if (wq > 1) {
        md->score.midgame -= 8;
        md->score.endgame -= 12;
    }
    if (bq > 1) {
        md->score.midgame += 8;
        md->score.endgame += 12;
    }

    // Pawn bonuses
    int material_adjust = 0;
    material_adjust += wn * 3 * (wp - 4);
    material_adjust -= bn * 3 * (bp - 4);
    material_adjust += wb * 2 * (wp - 4);
    material_adjust -= bb * 2 * (bp - 4);
    material_adjust += wr * (-3) * (wp - 4);
    material_adjust -= br * (-3) * (bp - 4);
    material_adjust += 10 * (b_minor - w_minor);
    material_adjust += 10 * (b_major - w_major);
    md->score.midgame += material_adjust;
    md->score.endgame += material_adjust;

    // Recognize specific material combinations where we want to do separate
    // scaling or scoring.
    md->eg_type = EG_NONE;
    md->scale[WHITE] = md->scale[BLACK] = 1024;
    if (w_all + b_all == 0) {
        md->eg_type = EG_DRAW;
    } else if (w_all + b_all == 1) {
        if (wp) {
            md->eg_type = EG_KPK;
            md->strong_side = WHITE;
        } else if (bp) {
            md->eg_type = EG_KPK;
            md->strong_side = BLACK;
        } else if (wq || wr) {
            md->eg_type = EG_WIN;
            md->strong_side = WHITE;
        } else if (bq || br) {
            md->eg_type = EG_WIN;
            md->strong_side = BLACK;
        } else {
            md->eg_type = EG_DRAW;
        } 
    } else if (w_all == 0) {
        if (b_piece > 2) {
            md->eg_type = EG_WIN;
            md->strong_side = BLACK;
        } else if (b_piece == 2) {
            if (bn == 2) {
                md->eg_type = EG_DRAW;
            } else if (bb && bn) {
                md->eg_type = EG_KBNK;
                md->strong_side = BLACK;
            } else {
                md->eg_type = EG_WIN;
                md->strong_side = BLACK;
            }
        }
    } else if (b_all == 0) {
        if (w_piece > 2) {
            md->eg_type = EG_WIN;
            md->strong_side = WHITE;
        } else if (w_piece == 2) {
            if (wn == 2) {
                md->eg_type = EG_DRAW;
            } else if (wb && wn) {
                md->eg_type = EG_KBNK;
                md->strong_side = WHITE;
            } else {
                md->eg_type = EG_WIN;
                md->strong_side = WHITE;
            }
        }
    }
    
    // Endgame scaling factors
    if (md->eg_type == EG_WIN) {
        md->scale[md->strong_side^1] = 0;
    } else if (md->eg_type == EG_DRAW) {
        md->scale[BLACK] = md->scale[WHITE] = 0;
        return;
    }

    // It's hard to win if you don't have any pawns, or if you only have one
    // and your opponent can trade it for a piece without leaving mating
    // material. Bishops tend to be better than knights in this scenario.
    if (!wp) {
        if (w_piece == 1) {
            md->scale[WHITE] = 0;
        } else if (w_piece == 2 && wn == 2) {
            md->scale[WHITE] = 32;
        } else if (w_piece - b_piece < 2 && w_major < 3) {
            md->scale[WHITE] = 128;
        } else if (w_piece == 2 && wb == 2) {
            md->scale[WHITE] = 768;
        } else if (!w_major) md->scale[WHITE] = 512;
    } else if (wp == 1 && b_piece) {
        if (w_piece == 1 || (w_piece == 2 && wn == 2)) {
            md->scale[WHITE] = 256;
        } else if (w_piece - b_piece + (b_major == 0) < 1 && w_major < 3) {
            md->scale[WHITE] = 512;
        }
    }

    if (!bp) {
        if (b_piece == 1) {
            md->scale[BLACK] = 0;
        } else if (b_piece == 2 && bn == 2) {
            md->scale[BLACK] = 32;
        } else if (b_piece - w_piece < 2 && b_major < 3) {
            md->scale[BLACK] = 128;
        } else if (b_piece == 2 && bb == 2) {
            md->scale[BLACK] = 768;
        } else if(!b_major) md->scale[BLACK] = 512;
    } else if (bp == 1 && w_piece) {
        if (b_piece == 1 || (b_piece == 2 && bn == 2)) {
            md->scale[BLACK] = 256;
        } else if (b_piece - w_piece + (w_major == 0) < 1 && b_major < 3) {
            md->scale[BLACK] = 512;
        }
    }
}
static pawn_data_t* pawn_table = NULL;
/*
 * Look up the material data for the given position.
 */
material_data_t* get_material_data(const position_t* pos)
{
    material_data_t* md = &material_table[pos->material_hash % num_buckets];
    if (md->key == pos->material_hash) {
        material_hash_stats.hits++;
        return md;
    } else if (md->key != 0) {
        material_hash_stats.evictions++;
    } else {
        material_hash_stats.misses++;
        material_hash_stats.occupied++;
    }
    compute_material_data(pos, md);
    md->key = pos->material_hash;
    return md;
}
static struct {
    int misses;
    int hits;
    int occupied;
    int evictions;
} pawn_hash_stats;
static pawn_data_t* get_pawn_data(const position_t* pos)
{
    pawn_data_t* pd = &pawn_table[pos->pawn_hash % num_buckets];
    if (pd->key == pos->pawn_hash) pawn_hash_stats.hits++;
    else if (pd->key != 0) pawn_hash_stats.evictions++;
    else {
        pawn_hash_stats.misses++;
        pawn_hash_stats.occupied++;
    }
    return pd;
}
static const int isolation_penalty[2][8] = {
    { 6, 6, 6, 8, 8, 6, 6, 6 },
    { 8, 8, 8, 8, 8, 8, 8, 8 }
};
static const int open_isolation_penalty[2][8] = {
    { 14, 14, 15, 16, 16, 15, 14, 14 },
    { 16, 17, 18, 20, 20, 18, 17, 16 }
};
static const int doubled_penalty[2][8] = {
    { 5, 5, 5, 6, 6, 5, 5, 5 },
    { 6, 7, 8, 8, 8, 8, 7, 6 }
};
static const int passed_bonus[2][8] = {
    { 0,  5, 10, 20, 60, 120, 200, 0 },
    { 0, 10, 20, 25, 75, 135, 225, 0 },
};
static const int candidate_bonus[2][8] = {
    { 0, 5,  5, 10, 20, 30, 0, 0 },
    { 0, 5, 10, 15, 30, 45, 0, 0 },
};
static const int backward_penalty[2][8] = {
    { 6, 6, 6,  8,  8, 6, 6, 6 },
    { 8, 9, 9, 10, 10, 9, 9, 8 }
};
static const int unstoppable_passer_bonus[8] = {
    0, 500, 525, 550, 575, 600, 650, 0
};
static const int advanceable_passer_bonus[8] = {
    0, 20, 25, 30, 35, 40, 80, 0
};
static const int king_dist_bonus[8] = {
    0, 0, 5, 10, 15, 20, 25, 0
};
static const int connected_passer[2][8] = {
    { 0, 0, 1, 2,  5, 15, 20, 0},
    { 0, 0, 2, 5, 15, 40, 60, 0}
};
static const int connected_bonus[2] = { 5, 5 };
static const int passer_rook[2] = { 5, 15 };
static const int king_storm[0x80] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,-10,-10,-10,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0, -8, -8, -8,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  4,  4,  4,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  8,  8,  8,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0, 12, 12, 12,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0, 14, 16, 14,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};
static const int queen_storm[0x80] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -10,-10,-10, -5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   -8, -8, -8, -4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    4,  4,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    8,  8,  8,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   12, 12, 12,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   14, 16, 14,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};
static const int central_space[0x80] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  2,  4,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  2,  4,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  1,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};
bitboard_t set_mask[64];
bitboard_t clear_mask[64];
bitboard_t in_front_mask[2][64];
bitboard_t outpost_mask[2][64];
bitboard_t passed_mask[2][64];
bitboard_t rank_mask[8] = {
    RANK_1_BB, RANK_2_BB, RANK_3_BB, RANK_4_BB,
    RANK_5_BB, RANK_6_BB, RANK_7_BB, RANK_8_BB
};
bitboard_t file_mask[8] = {
    FILE_A_BB, FILE_B_BB, FILE_C_BB, FILE_D_BB,
    FILE_E_BB, FILE_F_BB, FILE_G_BB, FILE_H_BB
};
bitboard_t neighbor_file_mask[8] = {
    FILE_B_BB, FILE_A_BB|FILE_C_BB, FILE_B_BB|FILE_D_BB, FILE_C_BB|FILE_E_BB,
    FILE_D_BB|FILE_F_BB, FILE_E_BB|FILE_G_BB, FILE_F_BB|FILE_H_BB, FILE_G_BB
};
/*
 * Identify and record the position of all passed pawns. Analyze pawn structure
 * features, such as isolated and doubled pawns, and assign a pawn structure
 * score (which does not account for passers). This information is stored in
 * the pawn hash table, to prevent re-computation.
 */
pawn_data_t* analyze_pawns(const position_t* pos)
{
    pawn_data_t* pd = get_pawn_data(pos);
    if (pd->key == pos->pawn_hash) return pd;

    // Zero everything out and create pawn bitboards.
    memset(pd, 0, sizeof(pawn_data_t));
    pd->key = pos->pawn_hash;
    square_t sq, to;
    for (color_t color=WHITE; color<=BLACK; ++color) {
        for (int i=0; pos->pawns[color][i] != INVALID_SQUARE; ++i) {
            set_sq_bit(pd->pawns_bb[color], pos->pawns[color][i]);
        }
    }

    // Create outpost bitboard and analyze pawns.
    for (color_t color=WHITE; color<=BLACK; ++color) {
        pd->num_passed[color] = 0;
        int push = pawn_push[color];
        const piece_t pawn = create_piece(color, PAWN);
        const piece_t opp_pawn = create_piece(color^1, PAWN);
        bitboard_t our_pawns = pd->pawns_bb[color];
        bitboard_t their_pawns = pd->pawns_bb[color^1];

        for (int ind=0; ind<64; ++ind) {
            // Fill in mask of outpost squares.
            sq = index_to_square(ind);
            if (!(outpost_mask[color][ind] & their_pawns)) {
                set_bit(pd->outposts_bb[color], ind);
            }
            if (pos->board[sq] != pawn) continue;

            file_t file = square_file(sq);
            rank_t rank = square_rank(sq);
            rank_t rrank = relative_rank[color][rank];

            // Passed pawns and passed pawn candidates.
            bool passed = !(passed_mask[color][ind] & their_pawns);
            if (passed) {
                set_bit(pd->passed_bb[color], ind);
                pd->passed[color][pd->num_passed[color]++] = sq;
                pd->score[color].midgame += passed_bonus[0][rrank];
                pd->score[color].endgame += passed_bonus[1][rrank];
            } else {
                // Candidate passed pawns (one enemy pawn one file away).
                // TODO: this condition could be more sophisticated.
                int blockers = 0;
                for (to = sq + push;
                        pos->board[to] != OUT_OF_BOUNDS; to += push) {
                    if (pos->board[to-1] == opp_pawn) ++blockers;
                    if (pos->board[to] == opp_pawn) blockers = 2;
                    if (pos->board[to+1] == opp_pawn) ++blockers;
                }
                if (blockers < 2) {
                    pd->score[color].midgame += candidate_bonus[0][rrank];
                    pd->score[color].endgame += candidate_bonus[1][rrank];
                }
            }

            // Isolated pawns.
            bool isolated = (neighbor_file_mask[file] & our_pawns) == 0;
            bool open = (in_front_mask[color][ind] & their_pawns) == 0;
            if (isolated) {
                if (open) {
                    pd->score[color].midgame -= open_isolation_penalty[0][file];
                    pd->score[color].endgame -= open_isolation_penalty[1][file];
                } else {
                    pd->score[color].midgame -= isolation_penalty[0][file];
                    pd->score[color].endgame -= isolation_penalty[1][file];
                }
            }

            // Pawn storm scores. Only used in opposite-castling positions.
            int storm = 1.5*king_storm[sq ^ (0x70*color)];
            if (storm && (passed_mask[color][ind] &
                        (~file_mask[file]) & their_pawns)) storm += storm/2;
            if (storm && open) storm += storm/2;
            pd->kingside_storm[color] += storm;
            storm = 1.5*queen_storm[sq ^ (0x70*color)];
            if (storm && (passed_mask[color][ind] &
                        (~file_mask[file]) & their_pawns)) storm += storm/2;
            if (storm && open) storm += storm/2;
            pd->queenside_storm[color] += storm;

            // Doubled pawns.
            bool doubled = (in_front_mask[color^1][ind] & our_pawns) != 0;
            if (doubled) {
                pd->score[color].midgame -= doubled_penalty[0][file];
                pd->score[color].endgame -= doubled_penalty[1][file];
            }

            // Connected pawns.
            bool connected = neighbor_file_mask[file] & our_pawns &
                (rank_mask[rank] | rank_mask[rank + (color == WHITE ? 1:-1)]);
            if (connected) {
                pd->score[color].midgame += connected_bonus[0];
                pd->score[color].endgame += connected_bonus[1];
            }

            // Space bonus for connected advanced central pawns.
            if (connected) pd->score[color].midgame +=
                central_space[sq ^ (0x70*color)];

            // Backward pawns (unsupportable by pawns, can't advance).
            // TODO: a simpler formulation would be nice.
            if (!passed && !isolated && !connected &&
                    pos->board[sq+push-1] != opp_pawn &&
                    pos->board[sq+push+1] != opp_pawn) {
                bool backward = true;
                for (to = sq; pos->board[to] != OUT_OF_BOUNDS; to -= push) {
                    if (pos->board[to-1] == pawn || pos->board[to+1] == pawn) {
                        backward = false;
                        break;
                    }
                }
                if (backward) {
                    for (to = sq + 2*push; pos->board[to] != OUT_OF_BOUNDS;
                            to += push) {
                        if (pos->board[to-1] == opp_pawn ||
                                pos->board[to+1] == opp_pawn) break;
                        if (pos->board[to-1] == pawn ||
                                pos->board[to+1] == pawn) {
                            backward = false;
                            break;
                        }
                    }
                    if (backward) {
                        pd->score[color].midgame -= backward_penalty[0][file];
                        pd->score[color].endgame -= backward_penalty[1][file];
                    }
                }
            }
        }

#define square_is_outpost(pd, sq, side) \
    (sq_bit_is_set((pd)->outposts_bb[side], (sq)))
#define file_is_half_open(pd, file, side) \
    (((pd)->pawns_bb[side] & file_mask[file]) == EMPTY_BB)
#define pawn_is_passed(pd, sq, side) \
    (sq_bit_is_set((pd)->passed_bb[side], (sq)))
        // Penalty for multiple pawn islands.
        int islands = 0;
        bool on_island = false;
        for (file_t f = FILE_A; f <= FILE_H; ++f) {
            if (!file_is_half_open(pd, f, color)) {
                if (!on_island) {
                    on_island = true;
                    islands++;
                }
            } else on_island = false;
        }
        if (islands) --islands;
        pd->score[color].midgame -= 2 * islands;
        pd->score[color].endgame -= 4 * islands;
    }
    return pd;
}
/*
 * Retrieve (and calculate if necessary) the pawn data associated with |pos|,
 * and use it to determine the overall pawn score for the given position. The
 * pawn data is also used as an input to other evaluation functions.
 */
/*
 * Count all attackers and defenders of a square to determine whether or not
 * a capture is advantageous. Captures with a positive static eval are
 * favorable. Note: this implementation does not account for pinned pieces.
 */
int static_exchange_eval(const position_t* pos, move_t move)
{
    square_t attacker_sq = get_move_from(move);
    square_t attacked_sq = get_move_to(move);
    piece_t attacker = get_move_piece(move);
    piece_t captured = get_move_capture(move);
    square_t attacker_sqs[2][16];
    int num_attackers[2] = { 0, 0 };
    int initial_attacker[2] = { 0, 0 };
    int index;

    // Find all the pieces that could be attacking.
    // Pawns are handled separately, because there are potentially a lot of
    // them and only a few squares they could attack from.
    if (pos->board[attacked_sq + NW] == BP && attacked_sq + NW != attacker_sq) {
        index = num_attackers[BLACK]++;
        attacker_sqs[BLACK][index] = attacked_sq + NW;
    }
    if (pos->board[attacked_sq + NE] == BP && attacked_sq + NE != attacker_sq) {
        index = num_attackers[BLACK]++;
        attacker_sqs[BLACK][index] = attacked_sq + NE;
    }
    if (pos->board[attacked_sq + SW] == WP && attacked_sq + SW != attacker_sq) {
        index = num_attackers[WHITE]++;
        attacker_sqs[WHITE][index] = attacked_sq + SW;
    }
    if (pos->board[attacked_sq + SE] == WP && attacked_sq + SE != attacker_sq) {
        index = num_attackers[WHITE]++;
        attacker_sqs[WHITE][index] = attacked_sq + SE;
    }
    
    for (color_t side=WHITE; side<=BLACK; ++side) {
        square_t from, att_sq;
        piece_t piece;
        square_t att_dir;
        for (int i=0; pos->pieces[side][i] != INVALID_SQUARE; ++i) {
            from = pos->pieces[side][i];
            piece = pos->board[from];
            if (from == attacker_sq) continue;
            if (!possible_attack(from, attacked_sq, piece)) continue;
            piece_type_t type = piece_type(piece);
            switch (type) {
                case KING:
                case KNIGHT:
                    attacker_sqs[side][num_attackers[side]++] = from;
                    break;
                case BISHOP:
                case ROOK:
                case QUEEN:
                    att_dir = (square_t)direction(from, attacked_sq);
                    for (att_sq = from + att_dir; att_sq != attacked_sq &&
                            pos->board[att_sq] == EMPTY; att_sq += att_dir) {}
                    if (att_sq == attacked_sq) {
                        attacker_sqs[side][num_attackers[side]++] = from;
                    }
                    break;
                default: assert(false);
            }
        }
    }

    // At this point, all unblocked attackers other than |attacker| have been
    // added to |attackers|. Now play out all possible captures in order of
    // increasing piece value (but starting with |attacker|) while alternating
    // colors. Whenever a capture is made, add any x-ray attackers that removal
    // of the piece would reveal.
    color_t side = piece_color(attacker);
    initial_attacker[side] = 1;
    int gain[32] = { material_value(captured) };
    int gain_index = 1;
    int capture_value = material_value(attacker);
    bool initial_attack = true;
    while (num_attackers[side] + initial_attacker[side] + 1 || initial_attack) {
        initial_attack = false;
        // add in any new x-ray attacks
        const attack_data_t* att_data =
            &get_attack_data(attacker_sq, attacked_sq);
        if (att_data->possible_attackers & Q_FLAG) {
            square_t sq;
            for (sq=attacker_sq - att_data->relative_direction;
                    pos->board[sq] == EMPTY;
                    sq -= att_data->relative_direction) {}
            if (pos->board[sq] != OUT_OF_BOUNDS) {
                const piece_t xray_piece = pos->board[sq];
#define get_piece_flag(piece)       piece_flags[(piece)]
                if (att_data->possible_attackers & 
                        get_piece_flag(xray_piece)) {
                    const color_t xray_color = piece_color(xray_piece);
                    index = num_attackers[xray_color]++;
                    attacker_sqs[xray_color][index] = sq;
                }
            }
        }

        // score the capture under the assumption that it's defended.
        gain[gain_index] = capture_value - gain[gain_index - 1];
        ++gain_index;
        side ^= 1;

        // find the next lowest valued attacker
        int least_value = material_value(KING)+1;
        int att_index = -1;
        for (int i=0; i<num_attackers[side]; ++i) {
            capture_value = material_value(pos->board[attacker_sqs[side][i]]);
            if (capture_value < least_value) {
                least_value = capture_value;
                att_index = i;
            }
        }
        if (att_index == -1) {
            assert(num_attackers[side] == 0);
            break;
        }
        attacker = pos->board[attacker_sqs[side][att_index]];
        attacker_sq = attacker_sqs[side][att_index];
        index = --num_attackers[side];
        attacker_sqs[side][att_index] = attacker_sqs[side][index];
        capture_value = least_value;
        if (piece_type(attacker) == KING && num_attackers[side^1]) break;
    }

    // Now that gain array is set up, scan back through to get score.
    --gain_index;
    while (--gain_index) {
        gain[gain_index-1] = -gain[gain_index-1] < gain[gain_index] ?
            -gain[gain_index] : gain[gain_index-1];
    }
    return gain[0];
}
/*
 * If we only care about whether or not a move is losing, sometimes we don't
 * need a full static exchange eval and can bail out early.
 */
int static_exchange_sign(const position_t* pos, move_t move)
{
    piece_type_t attacker_type = piece_type(get_move_piece(move));
    piece_type_t captured_type = piece_type(get_move_capture(move));
    if (attacker_type == KING || attacker_type <= captured_type) return 1;
    return static_exchange_eval(pos, move);
}
int distance_data_storage[256];
const int* distance_data = distance_data_storage+128;
score_t pawn_score(const position_t* pos, pawn_data_t** pawn_data)
{
    pawn_data_t* pd = analyze_pawns(pos);
    if (pawn_data) *pawn_data = pd;
    int passer_bonus[2] = {0, 0};
    int eg_passer_bonus[2] = {0, 0};
    int storm_score[2] = {0, 0};
    file_t king_file[2] = { square_file(pos->pieces[WHITE][0]),
                            square_file(pos->pieces[BLACK][0]) };
    for (color_t side=WHITE; side<=BLACK; ++side) {
        const square_t push = pawn_push[side];
        piece_t our_pawn = create_piece(side, PAWN);
        for (int i=0; i<pd->num_passed[side]; ++i) {
            square_t passer = pd->passed[side][i];
            assert(pos->board[passer] == create_piece(side, PAWN));
            square_t target = passer + push;
            rank_t rank = relative_rank[side][square_rank(passer)];
            if (pos->num_pieces[side^1] == 1) {
                // Other side is down to king+pawns. Is this passer stoppable?
                // This measure is conservative, which is fine.
                int prom_dist = 8 - rank;
                if (rank == RANK_2) --prom_dist;
                if (pos->side_to_move == side) --prom_dist;
                square_t prom_sq = square_file(passer) + A8*side;
#define get_attack_data(from, to)   board_attack_data[(from)-(to)]
#define distance(from,to)           distance_data[(from)-(to)]
                if (distance(pos->pieces[side^1][0], prom_sq) > prom_dist) {
                    passer_bonus[side] += unstoppable_passer_bonus[rank];
                }
            }

            // Adjust endgame bonus based on king proximity
            eg_passer_bonus[side] += king_dist_bonus[rank] *
                (distance(target, pos->pieces[side^1][0]) -
                 distance(target, pos->pieces[side][0]));

            // Is the passer connected to another friendly pawn?
            if (pos->board[passer-1] == our_pawn ||
                    pos->board[passer+1] == our_pawn ||
                    pos->board[passer-push-1] == our_pawn ||
                    pos->board[passer-push+1] == our_pawn) {
                passer_bonus[side] += connected_passer[0][rank];
                eg_passer_bonus[side] += connected_passer[1][rank];
            }

            // Find rooks behind the passer.
            square_t sq;
            for (sq = passer - push; pos->board[sq] == EMPTY; sq -= push) {}
            if (pos->board[sq] == create_piece(side, ROOK)) {
                passer_bonus[side] += passer_rook[0];
                eg_passer_bonus[side] += passer_rook[1];
            } else if (pos->board[sq] == create_piece(side^1, ROOK)) {
                passer_bonus[side] -= passer_rook[0];
                eg_passer_bonus[side] -= passer_rook[1];
            }

            // Can the pawn advance without being captured?
            if (pos->board[target] == EMPTY) {
                move_t push = rank == RANK_7 ?
                    create_move_promote(passer, target,
                            create_piece(side, PAWN), EMPTY, QUEEN) :
                    create_move(passer, target,
                            create_piece(side, PAWN), EMPTY);
                if (static_exchange_sign(pos, push) >= 0) {
                    passer_bonus[side] += advanceable_passer_bonus[rank];
                }
            }
        }
        // Apply pawn storm bonuses
        if (king_file[side] < FILE_E && king_file[side^1] > FILE_E) {
            storm_score[side] = pd->kingside_storm[side];
        } else if (king_file[side] > FILE_E && king_file[side^1] < FILE_E) {
            storm_score[side] = pd->queenside_storm[side];
        }
    }

    color_t side = pos->side_to_move;
    score_t score;
    score.midgame = pd->score[side].midgame + passer_bonus[side] -
        (pd->score[side^1].midgame + passer_bonus[side^1]) +
        storm_score[side] - storm_score[side^1];
    score.endgame = pd->score[side].endgame +
        passer_bonus[side] + eg_passer_bonus[side] -
        (pd->score[side^1].endgame +
         passer_bonus[side^1] + eg_passer_bonus[side^1]);
    return score;
}
/*
 * Combine two scores, scaling |addend| by the given factor.
 */
static void add_scaled_score(score_t* score, score_t* addend, int scale)
{
    score->midgame += addend->midgame * scale / 1024;
    score->endgame += addend->endgame * scale / 1024;
}

static const int trapped_bishop = 150;
static const int luft_penalty[2] = { 10, 10 };
/*
 * Find simple bad patterns that won't show up within reasonable search
 * depths. This is mostly trapped and blocked pieces.
 * TODO: trapped knight/rook patterns.
 * TODO: maybe merge this with eval_pieces so we have access to piece
 *       mobility information.
 */
score_t pattern_score(const position_t*pos)
{
    int s = 0;
    int eg_modifier = 0;
    if (pos->board[A2] == BB && pos->board[B3] == WP) s += trapped_bishop;
    if (pos->board[B1] == BB && pos->board[C2] == WP) s += trapped_bishop;
    if (pos->board[H2] == BB && pos->board[G3] == WP) s += trapped_bishop;
    if (pos->board[G1] == BB && pos->board[F2] == WP) s += trapped_bishop;

    if (pos->board[A7] == WB && pos->board[B6] == BP) s -= trapped_bishop;
    if (pos->board[B8] == WB && pos->board[C7] == BP) s -= trapped_bishop;
    if (pos->board[H7] == WB && pos->board[G6] == BP) s -= trapped_bishop;
    if (pos->board[G8] == WB && pos->board[F7] == BP) s -= trapped_bishop;

    square_t k = pos->pieces[WHITE][0];
    if (square_rank(k) == RANK_1) {
        bool luft = false;
        if (pos->board[k+N] != WP) luft = true;
        file_t f = square_file(k);
        if (!luft && f > FILE_A && pos->board[k+N-1] != WP) luft = true;
        if (!luft && f < FILE_H && pos->board[k+N+1] != WP) luft = true;
        if (!luft) {
            s -= luft_penalty[0];
            eg_modifier -= luft_penalty[1];
        }
    }
    k = pos->pieces[BLACK][0];
    if (square_rank(k) == RANK_8) {
        bool luft = false;
        if (pos->board[k+S] != BP) luft = true;
        file_t f = square_file(k);
        if (!luft && f > FILE_A && pos->board[k+S-1] != BP) luft = true;
        if (!luft && f < FILE_H && pos->board[k+S+1] != BP) luft = true;
        if (!luft) {
            s += luft_penalty[0];
            eg_modifier += luft_penalty[1];
        }
    }

    if (pos->side_to_move == BLACK) {
        s *= -1;
        eg_modifier *= -1;
    }
    score_t score;
    score.midgame = s;
    score.endgame = s + eg_modifier;
    return score;
}
static const int color_table[2][17] = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0}, // white
    {1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // black
};
static const int knight_outpost[0x80] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  1,  4,  4,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  2,  4,  5,  5,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  3,  6,  9,  9,  6,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  1,  3,  4,  4,  3,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};
static const int bishop_outpost[0x80] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    1,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    3,  5,  6,  6,  6,  5,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    3,  5,  6,  6,  6,  5,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    1,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};
static int outpost_score(const position_t* pos, square_t sq, piece_type_t type)
{
    color_t side = piece_color(pos->board[sq]);
    int bonus = type == KNIGHT ? knight_outpost[sq ^ (0x70*side)] : bishop_outpost[sq ^ (0x70*side)];
    int score = bonus;
    if (bonus) {
        // An outpost is better when supported by pawns.
        piece_t our_pawn = create_piece(side, PAWN);
        if (pos->board[sq - pawn_push[side] - 1] == our_pawn ||
                pos->board[sq - pawn_push[side] + 1] == our_pawn) {
            score += bonus/2;
            // Even better if an opposing knight/bishop can't capture it.
            // TODO: take care of the case where there's one opposing bishop
            // that's the wrong color. The position data structure needs to
            // be modified a little to make this efficient, or I need to pull
            // out bishop color info before doing outposts.
            piece_t their_knight = create_piece(side^1, KNIGHT);
            piece_t their_bishop = create_piece(side^1, BISHOP);
            if (pos->piece_count[their_knight] == 0 &&
                    pos->piece_count[their_bishop] == 0) {
                score += bonus;
            }
        }
    }
    return score;
}
static const int rook_on_7[2] = { 20, 40 };
static const int rook_half_open_file_bonus[2] = { 10, 10 };
static const int rook_open_file_bonus[2] = { 20, 10 };
static const int mobility_score_table[2][8][32] = {
    { // midgame
        {0},
        {0, 4},
        {-8, -4, 0, 4, 8, 12, 16, 18, 20},
        {-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 40, 40, 40, 40},
        {-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20},
        {-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7,
            -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        {0}, {0}
    },
    { // endgame
        {0},
        {0, 12},
        {-8, -4, 0, 4, 8, 12, 16, 18, 20},
        {-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 40, 40, 40, 40},
        {-10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50},
        {-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12,
            14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42},
        {0}, {0}
    },
};
/*
 * Compute the number of squares each non-pawn, non-king piece could move to,
 * and assign a bonus or penalty accordingly. Also assign miscellaneous
 * bonuses based on outpost squares, open files, etc.
 */
score_t pieces_score(const position_t* pos, pawn_data_t* pd)
{
    score_t score;
    int mid_score[2] = {0, 0};
    int end_score[2] = {0, 0};
    rank_t king_rank[2] = { relative_rank[WHITE]
                                [square_rank(pos->pieces[WHITE][0])],
                            relative_rank[BLACK]
                                [square_rank(pos->pieces[BLACK][0])] };
    color_t side;
    for (side=WHITE; side<=BLACK; ++side) {
        const int* mobile = color_table[side];
        square_t from, to;
        piece_t piece;
        for (int i=1; pos->pieces[side][i] != INVALID_SQUARE; ++i) {
            from = pos->pieces[side][i];
            piece = pos->board[from];
            piece_type_t type = piece_type(piece);
            int ps = 0;
            switch (type) {
                case KNIGHT:
                    ps += mobile[pos->board[from-33]];
                    ps += mobile[pos->board[from-31]];
                    ps += mobile[pos->board[from-18]];
                    ps += mobile[pos->board[from-14]];
                    ps += mobile[pos->board[from+14]];
                    ps += mobile[pos->board[from+18]];
                    ps += mobile[pos->board[from+31]];
                    ps += mobile[pos->board[from+33]];
                    if (square_is_outpost(pd, from, side)) {
                        int bonus = outpost_score(pos, from, KNIGHT);
                        mid_score[side] += bonus;
                        end_score[side] += bonus;
                    }
                    break;
                case BISHOP:
                    for (to=from-17; pos->board[to]==EMPTY; to-=17, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from-15; pos->board[to]==EMPTY; to-=15, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+15; pos->board[to]==EMPTY; to+=15, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+17; pos->board[to]==EMPTY; to+=17, ++ps) {}
                    ps += mobile[pos->board[to]];
                    if (square_is_outpost(pd, from, side)) {
                        int bonus = outpost_score(pos, from, BISHOP);
                        mid_score[side] += bonus;
                        end_score[side] += bonus;
                    }
                    break;
                case ROOK:
                    for (to=from-16; pos->board[to]==EMPTY; to-=16, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from-1; pos->board[to]==EMPTY; to-=1, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+1; pos->board[to]==EMPTY; to+=1, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+16; pos->board[to]==EMPTY; to+=16, ++ps) {}
                    ps += mobile[pos->board[to]];
                    int rrank = relative_rank[side][square_rank(from)];
                    if (rrank == RANK_7 && king_rank[side^1] == RANK_8) {
                        mid_score[side] += rook_on_7[0];
                        end_score[side] += rook_on_7[1];
                    }
                    file_t file = square_file(from);
                    if (file_is_half_open(pd, file, side)) {
                        mid_score[side] += rook_half_open_file_bonus[0];
                        end_score[side] += rook_half_open_file_bonus[1];
                        if (file_is_half_open(pd, file, side^1)) {
                            mid_score[side] += rook_open_file_bonus[0];
                            end_score[side] += rook_open_file_bonus[1];
                        }
                    }
                    break;
                case QUEEN:
                    for (to=from-17; pos->board[to]==EMPTY; to-=17, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from-15; pos->board[to]==EMPTY; to-=15, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+15; pos->board[to]==EMPTY; to+=15, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+17; pos->board[to]==EMPTY; to+=17, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from-16; pos->board[to]==EMPTY; to-=16, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from-1; pos->board[to]==EMPTY; to-=1, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+1; pos->board[to]==EMPTY; to+=1, ++ps) {}
                    ps += mobile[pos->board[to]];
                    for (to=from+16; pos->board[to]==EMPTY; to+=16, ++ps) {}
                    ps += mobile[pos->board[to]];
                    if (relative_rank[side][square_rank(from)] == RANK_7 &&
                            king_rank[side^1] == RANK_8) {
                        mid_score[side] += rook_on_7[0] / 2;
                        end_score[side] += rook_on_7[1] / 2;
                    }
                    break;
                default: assert(false);
            }
            mid_score[side] += mobility_score_table[0][type][ps];
            end_score[side] += mobility_score_table[1][type][ps];
        }
    }
    side = pos->side_to_move;
    score.midgame = mid_score[side] - mid_score[side^1];
    score.endgame = end_score[side] - end_score[side^1];
    return score;
}

const int shield_value[2][17] = {
    { 0, 8, 2, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 2, 4, 1, 1, 0, 0, 0 },
};
/*
 * Give some points for pawns directly in front of your king.
 */
static int king_shield_score(const position_t* pos, color_t side, square_t king)
{
    int s = 0;
    int push = pawn_push[side];
    s += shield_value[side][pos->board[king-1]] * 2;
    s += shield_value[side][pos->board[king+1]] * 2;
    s += shield_value[side][pos->board[king+push-1]] * 4;
    s += shield_value[side][pos->board[king+push]] * 6;
    s += shield_value[side][pos->board[king+push+1]] * 4;
    s += shield_value[side][pos->board[king+2*push-1]];
    s += shield_value[side][pos->board[king+2*push]] * 2;
    s += shield_value[side][pos->board[king+2*push+1]];
    return s;
}
/*
 * Compute the overall balance of king safety offered by pawn shields.
 */
static void evaluate_king_shield(const position_t* pos, int score[2])
{
    score[WHITE] = score[BLACK] = 0;
    int oo_score[2] = {0, 0};
    int ooo_score[2] = {0, 0};
    int castle_score[2] = {0, 0};
    square_t wk = pos->pieces[WHITE][0];
    if (pos->piece_count[BQ]) {
        score[WHITE] = king_shield_score(pos, WHITE, wk);
        if (has_oo_rights(pos, WHITE)) {
            oo_score[WHITE] = king_shield_score(pos, WHITE, G1);
        }
        if (has_ooo_rights(pos, WHITE)) {
            ooo_score[WHITE] = king_shield_score(pos, WHITE, C1);
        }
        castle_score[WHITE] = MAX(score[WHITE],
            MAX(oo_score[WHITE], ooo_score[WHITE]));
    }
    square_t bk = pos->pieces[BLACK][0];
    if (pos->piece_count[WQ]) {
        score[BLACK] = king_shield_score(pos, BLACK, bk);
        if (has_oo_rights(pos, BLACK)) {
            oo_score[BLACK] = king_shield_score(pos, BLACK, G8);
        }
        if (has_ooo_rights(pos, BLACK)) {
            ooo_score[BLACK] = king_shield_score(pos, BLACK, C8);
        }
        castle_score[BLACK] = MAX(score[BLACK],
                MAX(oo_score[BLACK], ooo_score[BLACK]));
    }
    score[WHITE] = (score[WHITE] + castle_score[WHITE])/2;
    score[BLACK] = (score[BLACK] + castle_score[BLACK])/2;
}

// For each (from, to) pair, which pieces can attack squares 1 away from to?
piece_flag_t near_attack_data_storage[256];
const piece_flag_t* near_attack_data = near_attack_data_storage + 128;
#define near_attack(from, to, piece) \
    ((near_attack_data[(from)-(to)] & piece_flags[(piece)]) != 0)
square_t near_attack_deltas[16][256][4];
/*
 * Is a the piece on |from| attacking a square adjacent to |target|?
 */
bool piece_attacks_near(const position_t* pos, square_t from, square_t target)
{
    piece_t p = pos->board[from];
    if (near_attack(from, target, p)) {
        if (piece_slide_type(p) == NO_SLIDE) return true;
        int delta;
#define INVALID_DELTA   0xff
        for (int i=0; (delta = near_attack_deltas[p][128+from-target][i]) !=
                INVALID_DELTA; ++i) {
            square_t sq = from + delta;
            if (pos->board[sq] == OUT_OF_BOUNDS) continue;
            direction_t att_dir = direction(from, sq);
            square_t x = from;
            while (x != sq) {
                x += att_dir;
                if (x == sq) return true;
                if (pos->board[x] != EMPTY) break;
            }
        }
    }
    return false;
}
const int king_attack_score[16] = {
    0, 0, 16, 16, 32, 64, 0, 0, 0, 0, 16, 16, 32, 64, 0, 0
};
const int num_king_attack_scale[16] = {
    0, 0, 640, 800, 1120, 1200, 1280, 1280,
    1344, 1344, 1408, 1408, 1472, 1472, 1536, 1536
};
/*
 * Compute a measure of king safety given by the number and type of pieces
 * attacking a square adjacent to the king.
 */
static void evaluate_king_attackers(const position_t* pos,
        int shield_score[2],
        int score[2])
{
    static const int bad_shield = 28;
    for (color_t side = WHITE; side <= BLACK; ++side) {
        score[side] = 0;
        if (pos->piece_count[create_piece(side, QUEEN)] == 0) continue;
        const square_t opp_king = pos->pieces[side^1][0];
        int num_attackers = 0;
        for (int i=1; i<pos->num_pieces[side]; ++i) {
            const square_t attacker = pos->pieces[side][i];
            if (piece_attacks_near(pos, attacker, opp_king)) {
                score[side] += king_attack_score[pos->board[attacker]];
                num_attackers++;
            }
        }
        if (shield_score[side^1] <= bad_shield) num_attackers += 2;
        score[side] = score[side]*num_king_attack_scale[num_attackers]/1024;
    }
}

score_t evaluate_king_safety(const position_t* pos, eval_data_t* ed)
{
    (void)ed;
    int shield_score[2], attack_score[2];

    evaluate_king_shield(pos, shield_score);
    evaluate_king_attackers(pos, shield_score, attack_score);

#define shield_scale    1024
#define attack_scale    1024
    score_t phase_score;
    color_t side = pos->side_to_move;
    phase_score.midgame =
        (attack_score[side] - attack_score[side^1])*attack_scale/1024 +
        (shield_score[side] - shield_score[side^1])*shield_scale/1024;
    phase_score.endgame = 0;
    return phase_score;
}
static const int tempo_bonus[2] = { 9, 2 };
/*
 * Blend endgame and midgame values linearly according to |phase|.
 */
static int blend_score(score_t* score, int phase)
{
    return (phase*score->midgame + (MAX_PHASE-phase)*score->endgame)/MAX_PHASE;
}
/*
 * Is it still possible for |side| to win the game?
 */
bool can_win(const position_t* pos, color_t side)
{
    return !(pos->num_pawns[side] == 0 &&
            pos->material_eval[side] < ROOK_VAL + KING_VAL);
}
/*
 * Print a breakdown of the static evaluation of |pos|.
 */
void report_eval(const position_t* pos)
{
    eval_data_t ed_storage;
    eval_data_t* ed = &ed_storage;
    color_t side = pos->side_to_move;
    score_t phase_score, component_score;
    ed->md = get_material_data(pos);

    int score = 0;
    int endgame_scale[2] = { ed->md->scale[WHITE], ed->md->scale[BLACK] };
    printf("scale\t\t(%5d, %5d)\n", endgame_scale[WHITE], endgame_scale[BLACK]);

    phase_score = ed->md->score;
    if (side == BLACK) {
        phase_score.midgame *= -1;
        phase_score.endgame *= -1;
    }
    printf("md_score\t(%5d, %5d)\n", phase_score.midgame, phase_score.endgame);
    phase_score.midgame += pos->piece_square_eval[side].midgame -
        pos->piece_square_eval[side^1].midgame;
    phase_score.endgame += pos->piece_square_eval[side].endgame -
        pos->piece_square_eval[side^1].endgame;
    printf("psq_score\t(%5d, %5d)\n", phase_score.midgame, phase_score.endgame);

#define pawn_scale      1024
#define pattern_scale   1024
#define pieces_scale    1024
#define safety_scale    1024

#define POLL_INTERVAL   0x3fff
#define MATE_VALUE      32000
#define DRAW_VALUE      0
#define MIN_MATE_VALUE (MATE_VALUE-1024)

    component_score = pawn_score(pos, &ed->pd);
    add_scaled_score(&phase_score, &component_score, pawn_scale);
    printf("pawn_score\t(%5d, %5d)\n", phase_score.midgame, phase_score.endgame);
    component_score = pattern_score(pos);
    add_scaled_score(&phase_score, &component_score, pattern_scale);
    printf("pattern_score\t(%5d, %5d)\n", phase_score.midgame, phase_score.endgame);
    component_score = pieces_score(pos, ed->pd);
    add_scaled_score(&phase_score, &component_score, pieces_scale);
    printf("pieces_score\t(%5d, %5d)\n", phase_score.midgame, phase_score.endgame);
    component_score = evaluate_king_safety(pos, ed);
    add_scaled_score(&phase_score, &component_score, safety_scale);
    printf("safety_score\t(%5d, %5d)\n", phase_score.midgame, phase_score.endgame);

    phase_score.midgame += tempo_bonus[0];
    phase_score.endgame += tempo_bonus[1];

    score = blend_score(&phase_score, ed->md->phase);
    score = (score * endgame_scale[score > 0 ? side : side^1]) / 1024;

    if (!can_win(pos, side)) score = MIN(score, DRAW_VALUE);
    if (!can_win(pos, side^1)) score = MAX(score, DRAW_VALUE);
    printf("final_score\t%5d\n", score);
}

const char glyphs[] = ".PNBRQK  pnbrqk";
/*
 * Convert a position to its FEN form.
 * (see wikipedia.org/wiki/Forsyth-Edwards_Notation)
 */
void position_to_fen_str(const position_t* pos, char* fen)
{
    int empty_run=0;
    for (square_t square=A8;; ++square) {
        if (empty_run && (pos->board[square] || !valid_board_index(square))) {
            *fen++ = empty_run + '0';
            empty_run = 0;
        }
        if (!valid_board_index(square)) {
            *fen++ = '/';
            square -= 0x19; // drop down to next rank
        } else if (pos->board[square]) {
            *fen++ = glyphs[pos->board[square]];
        } else empty_run++;
        if (square == H1) {
            if (empty_run) *fen++ = empty_run + '0';
            break;
        }
    }
    *fen++ = ' ';
    *fen++ = pos->side_to_move == WHITE ? 'w' : 'b';
    *fen++ = ' ';
    if (pos->castle_rights == CASTLE_NONE) *fen++ = '-';
    else {
        if (has_oo_rights(pos, WHITE)) *fen++ = 'K';
        if (has_ooo_rights(pos, WHITE)) *fen++ = 'Q';
        if (has_oo_rights(pos, BLACK)) *fen++ = 'k';
        if (has_ooo_rights(pos, BLACK)) *fen++ = 'q';
    }
    *fen++ = ' ';
    if (pos->ep_square != EMPTY && valid_board_index(pos->ep_square)) {
        *fen++ = square_file(pos->ep_square) + 'a';
        *fen++ = square_rank(pos->ep_square) + '1';
    } else *fen++ = '-';
    *fen++ = ' ';
    fen += sprintf(fen, "%d", pos->fifty_move_counter);
    *fen++ = ' ';
    fen += sprintf(fen, "%d", (pos->ply+1)/2);
    *fen = '\0';
}
/*
 * Print an ascii representation of the current board.
 */
void print_board(const position_t* pos, bool uci_prefix)
{
# if __WORDSIZE == 64
#  define __PRI64_PREFIX	"l"
#  define __PRIPTR_PREFIX	"l"
# else
#  define __PRI64_PREFIX	"ll"
#  define __PRIPTR_PREFIX
# endif
/* lowercase hexadecimal notation.  */
# define PRIx8		"x"
# define PRIx16		"x"
# define PRIx32		"x"
# define PRIx64		__PRI64_PREFIX "x"
    char fen_str[256];
    position_to_fen_str(pos, fen_str);
    if (uci_prefix) printf("info string ");
    printf("fen: %s\n", fen_str);
    if (uci_prefix) printf("info string ");
    printf("hash: %"PRIx64"\n", pos->hash);
    if (uci_prefix) printf("info string ");
    for (square_t sq = A8; sq != INVALID_SQUARE; ++sq) {
        if (!valid_board_index(sq)) {
            printf("\n");
            if (sq < 0x18) break;
            if (uci_prefix) printf("info string ");
            sq -= 0x19;
            continue;
        }
        printf("%c ", glyphs[pos->board[sq]]);
    }
    report_eval(pos);
}

/*
 * Convert an algebraic string representation of a square (e.g. A1, c6) to
 * a square_t.
 */
square_t coord_str_to_square(const char* alg_square)
{
    if (tolower(alg_square[0]) < 'a' || tolower(alg_square[0]) > 'h' ||
        alg_square[1] < '0' || alg_square[1] > '9') return EMPTY;
    return create_square(tolower(alg_square[0])-'a', alg_square[1]-'1');
}


/*
 * Set up the basic data structures of a position. Used internally by
 * set_position, but does not result in a legal board and should not be used
 * elsewhere.
 */
static void init_position(position_t* position)
{
    memset(position, 0, sizeof(position_t));
    position->board = position->_board_storage+64;
    for (int square=0; square<256; ++square) {
        position->_board_storage[square] = OUT_OF_BOUNDS;
    }
    for (int i=0; i<64; ++i) {
        int square = index_to_square(i);
        position->piece_index[square] = -1;
        position->board[square] = EMPTY;
    }

    for (color_t color=WHITE; color<=BLACK; ++color) {
        for (int index=0; index<32; ++index) {
            position->pieces[color][index] = INVALID_SQUARE;
        }
        for (int index=0; index<16; ++index) {
            position->pawns[color][index] = INVALID_SQUARE;
        }
    }
}

hashkey_t hash_position(const position_t* pos)
{
    hashkey_t hash = 0;
    for (square_t sq=A1; sq<=H8; ++sq) {
        if (!valid_board_index(sq) || !pos->board[sq]) continue;
        hash ^= piece_hash(pos->board[sq], sq);
    }
    hash ^= ep_hash(pos);
    hash ^= castle_hash(pos);
    hash ^= side_hash(pos);
    return hash;
}
hashkey_t hash_material(const position_t* pos)
{
    hashkey_t hash = 0;
    piece_t p = 0;
    for (piece_type_t pt = PAWN; pt<=KING; ++pt) {
        p = create_piece(WHITE, pt);
        for (int i=0; i<pos->piece_count[p]; ++i) hash ^= material_hash(p, i);
        p = create_piece(BLACK, pt);
        for (int i=0; i<pos->piece_count[p]; ++i) hash ^= material_hash(p, i);
    }
    return hash;
}
hashkey_t hash_pawns(const position_t* pos)
{
    hashkey_t hash = 0;
    for (square_t sq=A1; sq<=H8; ++sq) {
        if (!valid_board_index(sq) || !pos->board[sq]) continue;
        if (!piece_is_type(pos->board[sq], PAWN)) continue;
        hash ^= piece_hash(pos->board[sq], sq);
    }
    return hash;
}
/*
 * Set all hashes associated with a position to their correct values.
 */
void set_hash(position_t* pos)
{
    pos->hash = hash_position(pos);
    pos->material_hash = hash_material(pos);
    pos->pawn_hash = hash_pawns(pos);
}
/*
 * Given an FEN position description, set the given position to match it.
 * (see wikipedia.org/wiki/Forsyth-Edwards_Notation)
 */
char* set_position(position_t* pos, const char* fen)
{
    const char* orig_fen = fen;
    init_position(pos);
    king_home = E1;
    king_rook_home = H1;
    queen_rook_home = A1;

    // Read piece positions.
    for (square_t square=A8; square!=INVALID_SQUARE; ++fen, ++square) {
        if (isdigit(*fen)) {
            if (*fen == '0' || *fen == '9') {
                warn("Invalid FEN string");
                return (char*)fen;
            } else {
                square += *fen - '1';
            }
            continue;
        }
        switch (*fen) {
            case 'p': place_piece(pos, BP, square); break;
            case 'P': place_piece(pos, WP, square); break;
            case 'n': place_piece(pos, BN, square); break;
            case 'N': place_piece(pos, WN, square); break;
            case 'b': place_piece(pos, BB, square); break;
            case 'B': place_piece(pos, WB, square); break;
            case 'r': place_piece(pos, BR, square); break;
            case 'R': place_piece(pos, WR, square); break;
            case 'q': place_piece(pos, BQ, square); break;
            case 'Q': place_piece(pos, WQ, square); break;
            case 'k': place_piece(pos, BK, square); break;
            case 'K': place_piece(pos, WK, square); break;
            case '/': square -= 17 + square_file(square); break;
            case ' ': square = INVALID_SQUARE-1; break;
            case '\0':
            case '\n':set_hash(pos);
                      pos->is_check = find_checks(pos);
                      check_board_validity(pos);
                      return (char*)fen;
            default: warn("Illegal character in FEN string");
                     warn(orig_fen);
                     return (char*)fen;
        }
    }
    while (isspace(*fen)) ++fen;

    // Read whose turn is next.
    switch (tolower(*fen)) {
        case 'w': pos->side_to_move = WHITE; break;
        case 'b': pos->side_to_move = BLACK; break;
        default:  warn("Illegal side to move in FEN string");
                  return (char*)fen;
    }
    while (*fen && isspace(*(++fen))) {}
    pos->is_check = find_checks(pos);

    // Read castling rights. This is complicated by the need to support all
    // formats for castling flags in Chess960.
    while (*fen && !isspace(*fen)) {
        square_t sq;
        switch (*fen) {
            case 'q': add_ooo_rights(pos, BLACK);
                      for (sq=A8; pos->board[sq]!=BR && sq<=H8; ++sq) {}
                      if (pos->board[sq] == BR) {
                          queen_rook_home = square_file(sq);
                          king_home = square_file(pos->pieces[BLACK][0]);
                      } else warn("inconsistent castling rights");
                      break;
            case 'Q': add_ooo_rights(pos, WHITE);
                      for (sq=A1; pos->board[sq]!=WR && sq<=H1; ++sq) {}
                      if (pos->board[sq] == WR) {
                          queen_rook_home = sq;
                          king_home = pos->pieces[WHITE][0];
                      } else warn("inconsistent castling rights");
                      break;
            case 'k': add_oo_rights(pos, BLACK);
                      for (sq=H8; pos->board[sq]!=BR && sq>=A8; --sq) {}
                      if (pos->board[sq] == BR) {
                          king_rook_home = square_file(sq);
                          king_home = square_file(pos->pieces[BLACK][0]);
                      } else warn("inconsistent castling rights");
                      break;
            case 'K': add_oo_rights(pos, WHITE);
                      for (sq=H1; pos->board[sq]!=WR && sq>=A1; --sq) {}
                      if (pos->board[sq] == WR) {
                          king_rook_home = sq;
                          king_home = pos->pieces[WHITE][0];
                      } else warn("inconsistent castling rights");
                      break;
            case '-': break;
            default:
                // Chess960 castling flags.
                if (*fen >= 'A' && *fen <= 'H') {
                    king_home = pos->pieces[WHITE][0];
                    if (*fen - 'A' < king_home) {
                        add_ooo_rights(pos, WHITE);
                        queen_rook_home = *fen - 'A';
                    } else {
                        add_oo_rights(pos, WHITE);
                        king_rook_home = *fen - 'A';
                    }
                } else if (*fen >= 'a' && *fen <= 'h') {
                    king_home = square_file(pos->pieces[BLACK][0]);
                    if (*fen - 'a' < king_home) {
                        add_ooo_rights(pos, BLACK);
                        queen_rook_home = *fen - 'a';
                    } else {
                        add_oo_rights(pos, BLACK);
                        king_rook_home = *fen - 'a';
                    }
                } else {
                    // The fen string must have ended prematurely.
                    set_hash(pos);
                    check_board_validity(pos);
                    return (char*)fen;
                }
        }
        ++fen;
    }
    while (isspace(*fen)) ++fen;
    if (!*fen) {
        set_hash(pos);
        check_board_validity(pos);
        return (char*)fen;
    }

    // Read en-passant square
    if (*fen != '-') {
        square_t ep_sq = coord_str_to_square(fen++);
        piece_t pawn = create_piece(pos->side_to_move, PAWN);
        if (pos->board[ep_sq - pawn_push[pos->side_to_move] - 1] == pawn ||
                pos->board[ep_sq - pawn_push[pos->side_to_move] + 1] == pawn) {
            pos->ep_square = ep_sq;
        }
        if (*fen) ++fen;
    }
    while (*fen && isspace(*(++fen))) {}
    if (!*fen) {
        set_hash(pos);
        check_board_validity(pos);
        return (char*)fen;
    }

    // Read 50-move rule status and current move number.
    int consumed;
    if (sscanf(fen, "%d %d%n", &pos->fifty_move_counter, &pos->ply,
                &consumed)) fen += consumed;
    pos->ply = 0;
    set_hash(pos);
    check_board_validity(pos);
    return (char*)fen;
}


#define PLY                 1.0
#define MAX_SEARCH_PLY      127
#define depth_to_index(x)   ((int)(x))

#define MAX_HISTORY         1000000
#define MAX_HISTORY_INDEX   (16*64)
#define depth_to_history(d) ((d)*(d))
#define history_index(m)   \
    ((get_move_piece_type(m)<<6)|(square_to_index(get_move_to(m))))
#define HIST_BUCKETS    15


typedef enum {
    SEARCH_ABORTED, SEARCH_FAIL_HIGH, SEARCH_FAIL_LOW, SEARCH_EXACT
} search_result_t;

typedef struct {
    move_t pv[MAX_SEARCH_PLY+1];
    move_t killers[2];
    move_t mate_killer;
} search_node_t;

typedef struct {
    int transposition_cutoffs[MAX_SEARCH_PLY + 1];
    int nullmove_cutoffs[MAX_SEARCH_PLY + 1];
    int move_selection[HIST_BUCKETS + 1];
    int pv_move_selection[HIST_BUCKETS + 1];
    int razor_attempts[3];
    int razor_prunes[3];
    int root_fail_highs;
    int root_fail_lows;
    int egbb_hits;
} search_stats_t;
typedef struct {
    uint64_t nodes;
    move_t move;
    int score;
    int max_ply;
    int qsearch_score;
    move_t pv[MAX_SEARCH_PLY + 1];
} root_move_t;
typedef struct {
    float history[16*64]; // move indexed by piece type and destination square
    int success[16*64];
    int failure[16*64];
} history_t;
typedef enum {
    ENGINE_IDLE=0, ENGINE_PONDERING, ENGINE_THINKING, ENGINE_ABORTED
} engine_status_t;

typedef struct {
    struct timeval tv_start;
    int elapsed_millis;
    bool running;
} milli_timer_t;

typedef struct {
    position_t root_pos;
    search_stats_t stats;

    // search state info
    root_move_t root_moves[256];
    root_move_t* current_root_move;
    int best_score;
    int scores_by_iteration[MAX_SEARCH_PLY + 1];
    int root_indecisiveness;
    move_t pv[MAX_SEARCH_PLY + 1];
    search_node_t search_stack[MAX_SEARCH_PLY + 1];
    history_t history;
    uint64_t nodes_searched;
    uint64_t qnodes_searched;
    uint64_t pvnodes_searched;
    float current_depth;
    int current_move_index;
    bool resolving_fail_high;
    move_t obvious_move;
    engine_status_t engine_status;

    // when should we stop?
    milli_timer_t timer;
    uint64_t node_limit;
    float depth_limit;
    int time_limit;
    int time_target;
    int time_bonus;
    int mate_search; // TODO: implement me
    bool infinite;
} search_data_t;

/*
 * Create a copy of |src| in |dst|.
 */
void copy_position(position_t* dst, const position_t* src)
{
    check_board_validity(src);
    memcpy(dst, src, sizeof(position_t));
    dst->board = dst->_board_storage+64;
    check_board_validity(src);
    check_board_validity(dst);
}
/*
 * Initialize a timer.
 */
void init_timer(milli_timer_t* timer)
{
    timer->elapsed_millis = 0;
    timer->running = false;
}
/*
 * Zero out all search variables prior to starting a search. Leaves the
 * position and search options untouched.
 */
void init_search_data(search_data_t* data)
{
    position_t root_pos_copy;
    copy_position(&root_pos_copy, &data->root_pos);
    memset(data, 0, sizeof(search_data_t));
    copy_position(&data->root_pos, &root_pos_copy);
    data->engine_status = ENGINE_IDLE;
    init_timer(&data->timer);
}
void srandom_32(unsigned seed)
{
    srand(seed);
}
int32_t random_32(void)
{
    int r = rand();
    r <<= 16;
    r |= rand();
    return r;
}
static hashkey_t random_hashkey(void)
{
    // Sadly we only get 16 usable bits on Windows, so lots of stitching
    // is needed. This isn't performance-sensitive or cryptographical
    // though, so this should be fine.
    hashkey_t hash = random_32();
    hash <<= 32;
    hash |= random_32();
    return hash;
}

void init_hash(void)
{
    int i;
    srandom_32(1);
    hashkey_t* _piece_random = (hashkey_t*)&piece_random[0][0][0];
    hashkey_t* _castle_random = (hashkey_t*)&castle_random[0][0][0];
    hashkey_t* _enpassant_random = (hashkey_t*)&enpassant_random[0];

    for (i=0; i<2*7*64; ++i) _piece_random[i] = random_hashkey();
    for (i=0; i<64; ++i) _enpassant_random[i] = random_hashkey();
    for (i=0; i<2*2*2; ++i) _castle_random[i] = random_hashkey();
}
void clear_material_table(void)
{
    memset(material_table, 0, sizeof(material_data_t) * num_buckets);
    memset(&material_hash_stats, 0, sizeof(material_hash_stats));
}
void init_material_table(const int max_bytes)
{
    assert(max_bytes >= 1024);
    int size = sizeof(material_data_t);
    num_buckets = 1;
    while (size <= max_bytes >> 1) {
        size <<= 1;
        num_buckets <<= 1;
    }
    if (material_table != NULL) free(material_table);
    material_table = malloc(size);
    assert(material_table);
    clear_material_table();
}

search_data_t root_data;
#define FEN_STARTPOS "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
bool big_endian;
void init_bitboards(void)
{
    for (int sq=0; sq<64; ++sq) {
        set_mask[sq] = 1ull<<sq;
        clear_mask[sq] = ~set_mask[sq];
        int sq_rank = sq / 8;
        int sq_file = sq % 8;
        in_front_mask[WHITE][sq] = in_front_mask[BLACK][sq] = EMPTY_BB;
        outpost_mask[WHITE][sq] = outpost_mask[BLACK][sq] = EMPTY_BB;
        passed_mask[WHITE][sq] = passed_mask[BLACK][sq] = EMPTY_BB;
        for (int rank = sq_rank+1; rank<8; ++rank) {
            outpost_mask[WHITE][sq] |= rank_mask[rank];
            passed_mask[WHITE][sq] |= rank_mask[rank];
            in_front_mask[WHITE][sq] |= rank_mask[rank];
        }
        for (int rank = sq_rank-1; rank>=0; --rank) {
            outpost_mask[BLACK][sq] |= rank_mask[rank];
            passed_mask[BLACK][sq] |= rank_mask[rank];
            in_front_mask[BLACK][sq] |= rank_mask[rank];
        }
        bitboard_t passer_file_mask =
            file_mask[sq_file] | neighbor_file_mask[sq_file];
        bitboard_t outpost_file_mask = neighbor_file_mask[sq_file];
        outpost_mask[WHITE][sq] &= outpost_file_mask;
        outpost_mask[BLACK][sq] &= outpost_file_mask;
        passed_mask[WHITE][sq] &= passer_file_mask;
        passed_mask[BLACK][sq] &= passer_file_mask;
        in_front_mask[WHITE][sq] &= file_mask[sq_file];
        in_front_mask[BLACK][sq] &= file_mask[sq_file];
    }
}

static void add_near_attack(square_t target,
        square_t from,
        int delta,
        piece_t piece)
{
    near_attack_data_storage[128+from-target] |= get_piece_flag(piece);
    int i = 0;
    for (; near_attack_deltas[piece][128+from-target][i] != INVALID_DELTA;
            ++i) {
        if (near_attack_deltas[piece][128+from-target][i] == delta) return;
    }
    assert(near_attack_deltas[piece][128+from-target][i] == INVALID_DELTA);
    if (i<3) near_attack_deltas[piece][128+from-target][i] = delta;
    assert(near_attack_deltas[piece][128+from-target][0] != INVALID_DELTA);
}
void generate_attack_data(void)
{
    memset((char*)distance_data_storage, -1, sizeof(int)*256);
    for (square_t s1=A1; s1<=H8; ++s1) {
        if (!valid_board_index(s1)) continue;
        for (square_t s2=A1; s2<=H8; ++s2) {
            if (!valid_board_index(s2)) continue;
            distance_data_storage[128+s2-s1] = MAX(
                    abs(square_file(s2) - square_file(s1)),
                    abs(square_rank(s2) - square_rank(s1)));
        }
    }

    memset((char*)board_attack_data_storage, 0, sizeof(attack_data_t)*256);
    attack_data_t* mutable_attack_data = (attack_data_t*)board_attack_data;
    for (square_t from=A1; from<=H8; ++from) {
        if (!valid_board_index(from)) continue;
        for (piece_t piece=WP; piece<=BK; ++piece) {
            for (const direction_t* dir=piece_deltas[piece]; *dir; ++dir) {
                for (square_t to=from+*dir; valid_board_index(to); to+=*dir) {
                    mutable_attack_data[from-to].possible_attackers |=
                        get_piece_flag(piece);
                    mutable_attack_data[from-to].relative_direction = *dir;
                    if (piece_slide_type(piece) == NO_SLIDE) break;
                }
            }
        }
    }

    memset((char*)near_attack_data_storage, 0, sizeof(piece_flag_t)*256);
    for (int i=0; i<16; ++i)
        for (int j=0; j<256; ++j)
            for (int k=0; k<4; ++k)
                near_attack_deltas[i][j][k] = INVALID_DELTA;
    for (square_t target=A1; target<=H8; ++target) {
        if (!valid_board_index(target)) continue;
        for (square_t from=A1; from<=H8; ++from) {
            if (!valid_board_index(from)) continue;
            for (piece_t piece=WP; piece<=BK; ++piece) {
                for (const direction_t* dir=piece_deltas[piece]; *dir; ++dir) {
                    for (square_t to=from+*dir;
                            valid_board_index(to); to+=*dir) {
                        if (distance(target, to) == 1) {
                            add_near_attack(target, from, to-from, piece);
                            break;
                        }
                        if (piece_slide_type(piece) == NO_SLIDE) break;
                    }
                }
            }
        }
    }
}
const int eg_material_values[] = {
    0, EG_PAWN_VAL, EG_KNIGHT_VAL, EG_BISHOP_VAL,
    EG_ROOK_VAL, EG_QUEEN_VAL, EG_KING_VAL, 0,
    0, EG_PAWN_VAL, EG_KNIGHT_VAL, EG_BISHOP_VAL,
    EG_ROOK_VAL, EG_QUEEN_VAL, EG_KING_VAL, 0, 0
};

void init_eval(void)
{
    for (piece_t piece=WP; piece<=WK; ++piece) {
        for (square_t square=A1; square<=H8; ++square) {
            if (!valid_board_index(square)) continue;
            piece_square_values[piece][square] =
                piece_square_values[piece+BP-1][flip_square(square)];
            endgame_piece_square_values[piece][square] =
                endgame_piece_square_values[piece+BP-1][flip_square(square)];
        }
    }
    for (piece_t piece=WP; piece<=BK; ++piece) {
        if (piece > WK && piece < BP) continue;
        for (square_t square=A1; square<=H8; ++square) {
            if (!valid_board_index(square)) continue;
            piece_square_values[piece][square] += material_value(piece);
#define eg_material_value(piece)            eg_material_values[piece]
            endgame_piece_square_values[piece][square] +=
                eg_material_value(piece);
        }
    }
}

typedef enum {
    OPTION_CHECK,
    OPTION_SPIN,
    OPTION_COMBO,
    OPTION_BUTTON,
    OPTION_STRING
} uci_option_type_t;


typedef void(*option_handler)(void*, char*);
typedef struct {
    uci_option_type_t type;
    char name[128];
    char value[128];
    char vars[16][128];
    char default_value[128];
    int min;
    int max;
    void* address;
    option_handler handler;
} uci_option_t;
static int uci_option_count = 0;
static uci_option_t uci_options[128];

static uci_option_t* get_uci_option(const char* name)
{
    for (int i=0; i<uci_option_count; ++i) {
        int name_length = strlen(uci_options[i].name);
        if (!strncasecmp(name, uci_options[i].name, name_length)) {
            return &uci_options[i];
        }
    }
    return NULL;
}
void set_uci_option(char* command)
{
    while (isspace(*command)) ++command;
    uci_option_t* option = get_uci_option(command);
    if (!option) {
        printf("info string Could not recognize option string %s\n", command);
        return;
    }
    command = strcasestr(command, "value ");
    if (command) {
        command += 6;
        while(isspace(*command)) ++command;
    } else if (option->type != OPTION_BUTTON) {
        printf("info string Invalid option string\n");
        return;
    }
    option->handler(option, command);
}
static void add_uci_option(char* name,
        uci_option_type_t type,
        char* default_value,
        int min,
        int max,
        char** vars,
        void* address,
        option_handler handler)
{
    assert(uci_option_count < 128);
    uci_option_t* option = &uci_options[uci_option_count++];
    strcpy(option->name, name);
    strcpy(option->default_value, default_value);
    strcpy(option->value, default_value);
    option->type = type;
    option->min = min;
    option->max = max;
    option->handler = handler;
    option->vars[0][0] = '\0';
    option->address = address;
    int var_index=0;
    while (vars && vars[var_index]) {
        assert(var_index < 15);
        strcpy(option->vars[var_index], vars[var_index]);
        var_index++;
    }
    option->vars[var_index][0] = '\0';
    char option_command[256];
    sprintf(option_command, "%s value %s", option->name, option->default_value);
    set_uci_option(option_command);
}
typedef struct {
    hashkey_t key;
    move_t move;
    float depth;
    int16_t score;
    uint8_t age;
    uint8_t flags;
} transposition_entry_t;
static const int bucket_size = 4;
static transposition_entry_t* transposition_table = NULL;
static struct {
    uint64_t misses;
    uint64_t hits;
    uint64_t occupied;
    uint64_t alpha;
    uint64_t beta;
    uint64_t exact;
    uint64_t evictions;
    uint64_t collisions;
} hash_stats;
void clear_transposition_table(void)
{
    memset(transposition_table, 0,
            sizeof(transposition_entry_t)*bucket_size*num_buckets);
    memset(&hash_stats, 0, sizeof(hash_stats));
}
static const int generation_limit = 8;
static int generation;
static int age_score_table[8];
static void set_transposition_age(int age)
{
    assert(age >= 0 && age < generation_limit);
    generation = age;
    for (int i=0; i<generation_limit; ++i) {
        age = generation - i;
        if (age < 0) age += generation_limit;
        age_score_table[i] = age * 128;
    }
    memset(&hash_stats, 0, sizeof(hash_stats));
}
void init_transposition_table(const size_t max_bytes)
{
    assert(max_bytes >= 1024);
    size_t size = sizeof(transposition_entry_t) * bucket_size;
    num_buckets = 1;
    while (size <= max_bytes >> 1) {
        size <<= 1;
        num_buckets <<= 1;
    }
    if (transposition_table) free(transposition_table);
    transposition_table = malloc(size);
    assert(transposition_table);
    clear_transposition_table();
    set_transposition_age(0);
}
static void handle_hash(void* opt, char* value)
{
    uci_option_t* option = opt;
    int mbytes = 0;
    strncpy(option->value, value, 128);
    sscanf(value, "%d", &mbytes);
    if (mbytes < option->min || mbytes > option->max) {
        warn("Option value out of range, using default\n");
        sscanf(option->default_value, "%d", &mbytes);
    }
    init_transposition_table(mbytes * (1ull<<20));
}
static void handle_clear_hash(void* opt, char* value)
{
    (void) opt; (void) value;
    clear_transposition_table();
}
typedef move_t(*book_fn)(position_t*);
typedef struct {
    int multi_pv;
    int output_delay;
    bool use_book;
    bool book_loaded;
    book_fn probe_book;
    bool use_scorpio_bb;
    bool use_gtb;
    bool use_gtb_dtm;
    bool root_in_gtb;
    bool nonblocking_gtb;
    int gtb_cache_size;
    int gtb_scheme;
    int max_egtb_pieces;
    int verbosity;
    bool chess960;
    bool arena_castle;
    bool ponder;
} options_t;
options_t options;
static void default_handler(void* opt, char* value)
{
    uci_option_t* option = opt;
    if (value) strncpy(option->value, value, 128);
    if (option->address) {
        if (option->type == OPTION_CHECK) {
            bool val = strcasestr(option->value, "true") ? true : false;
            memcpy(option->address, &val, sizeof(bool));
        } else if (option->type == OPTION_SPIN) {
            int val;
            sscanf(option->value, "%d", &val);
            memcpy(option->address, &val, sizeof(int));
        } else assert(false);
    }
}
FILE* ctg_file = NULL;
FILE* cto_file = NULL;
typedef struct {
    int pad;
    int low;
    int high;
} page_bounds_t;
page_bounds_t page_bounds;
bool init_ctg_book(char* filename)
{
    int name_len = strlen(filename);
    assert(filename[name_len-3] == 'c' &&
            filename[name_len-2] == 't' &&
            filename[name_len-1] == 'g');
    char fbuf[1024];
    strcpy(fbuf, filename);
    if (ctg_file) {
        assert(cto_file);
        fclose(ctg_file);
        fclose(cto_file);
    }
    ctg_file = fopen(fbuf, "r");
    fbuf[name_len-1] = 'o';
    cto_file = fopen(fbuf, "r");
    fbuf[name_len-1] = 'b';
    FILE* ctb_file = fopen(fbuf, "r");
    fbuf[name_len-1] = 'g';
    if (!ctg_file || !cto_file || !ctb_file) {
        printf("info string Couldn't load book %s\n", fbuf);
        return false;
    }

#define my_ntohl(x) \
    (!big_endian ? \
    ((((uint32_t)(x) & 0xff000000) >> 24) | \
     (((uint32_t)(x) & 0x00ff0000) >>  8) | \
     (((uint32_t)(x) & 0x0000ff00) <<  8) | \
     (((uint32_t)(x) & 0x000000ff) << 24)) : \
     (x))
    // Read out upper and lower page limits.
    fread(&page_bounds, 12, 1, ctb_file);
    page_bounds.low = my_ntohl((uint32_t)page_bounds.low);
    page_bounds.high = my_ntohl((uint32_t)page_bounds.high);
    assert(page_bounds.low <= page_bounds.high);
    fclose(ctb_file);
    return true;
}
typedef struct {
    int num_moves;
    uint8_t moves[100];
    int total;
    int wins;
    int losses;
    int draws;
    int unknown1;
    int avg_rating_games;
    int avg_rating_score;
    int perf_rating_games;
    int perf_rating_score;
    int recommendation;
    int unknown2;
    int comment;
} ctg_entry_t;
typedef struct {
    uint8_t buf[64];
    int buf_len;
} ctg_signature_t;
const piece_t flip_piece[16] = {
    0, BP, BN, BB, BR, BQ, BK, 0,
    0, WP, WN, WB, WR, WQ, WK
};
static void append_bits_reverse(ctg_signature_t* sig,
        uint8_t bits,
        int bit_position,
        int num_bits)
{
    uint8_t * sig_byte = &sig->buf[bit_position/8];
    int offset = bit_position % 8;
    for (int i=offset; i<num_bits+offset; ++i, bits>>=1) {
        if (bits & 1) *sig_byte |= 1 << (7-(i%8));
        if (i%8 == 7) *(++sig_byte) = 0;
    }
}
static void position_to_ctg_signature(position_t* pos, ctg_signature_t* sig)
{
    // Note: initial byte is reserved for length and flags info
    memset(sig, 0, sizeof(ctg_signature_t));
    int bit_position = 8;
    uint8_t bits = 0, num_bits = 0;

    // The board is flipped if it's black's turn, and mirrored if the king is
    // on the queenside with no castling rights for either side.
    bool flip_board = pos->side_to_move == BLACK;
    color_t white = flip_board ? BLACK : WHITE;
    bool mirror_board = square_file(pos->pieces[white][0]) < FILE_E &&
        pos->castle_rights == 0;


#define flip_piece(p)                   (flip_piece[p])
    // For each board square, append the huffman bit sequence for its contents.
    for (int file=0; file<8; ++file) {
        for (int rank=0; rank<8; ++rank) {
            square_t sq = create_square(file, rank);
            if (flip_board) sq = mirror_rank(sq);
            if (mirror_board) sq = mirror_file(sq);
            piece_t piece = flip_board ?
                flip_piece[pos->board[sq]] :
                pos->board[sq];
            switch (piece) {
                case EMPTY: bits = 0x0; num_bits = 1; break;
                case WP: bits = 0x3; num_bits = 3; break;
                case BP: bits = 0x7; num_bits = 3; break;
                case WN: bits = 0x9; num_bits = 5; break;
                case BN: bits = 0x19; num_bits = 5; break;
                case WB: bits = 0x5; num_bits = 5; break;
                case BB: bits = 0x15; num_bits = 5; break;
                case WR: bits = 0xD; num_bits = 5; break;
                case BR: bits = 0x1D; num_bits = 5; break;
                case WQ: bits = 0x11; num_bits = 6; break;
                case BQ: bits = 0x31; num_bits = 6; break;
                case WK: bits = 0x1; num_bits = 6; break;
                case BK: bits = 0x21; num_bits = 6; break;
                default: assert(false);
            }
            append_bits_reverse(sig, bits, bit_position, num_bits);
            bit_position += num_bits;
        }
    }

    // Encode castling and en passant rights. These must sit flush at the end
    // of the final byte, so we also have to figure out how much to pad.
    int ep = -1;
    int flag_bit_length = 0;
    if (pos->ep_square) {
        ep = square_file(pos->ep_square);
        if (mirror_board) ep = 7 - ep;
        flag_bit_length = 3;
    }
    int castle = 0;
    if (has_oo_rights(pos, white)) castle += 4;
    if (has_ooo_rights(pos, white)) castle += 8;
    if (has_oo_rights(pos, white^1)) castle += 1;
    if (has_ooo_rights(pos, white^1)) castle += 2;
    if (castle) flag_bit_length += 4;
    uint8_t flag_bits = castle;
    if (ep != -1) {
        flag_bits <<= 3;
        for (int i=0; i<3; ++i, ep>>=1) if (ep&1) flag_bits |= (1<<(2-i));
    }

    //printf("\nflag bits: %d\n", flag_bits);
    //printf("bit_position: %d\n", bit_position%8);
    //printf("flag_bit_length: %d\n", flag_bit_length);

    // Insert padding so that flags fit at the end of the last byte.
    int pad_bits = 0;
    if (8-(bit_position % 8) < flag_bit_length) {
        //printf("padding byte\n");
        pad_bits = 8 - (bit_position % 8);
        append_bits_reverse(sig, 0, bit_position, pad_bits);
        bit_position += pad_bits;
    }

    pad_bits = 8 - (bit_position % 8) - flag_bit_length;
    if (pad_bits < 0) pad_bits += 8;
    //printf("padding %d bits\n", pad_bits);
    append_bits_reverse(sig, 0, bit_position, pad_bits);
    bit_position += pad_bits;
    append_bits_reverse(sig, flag_bits, bit_position, flag_bit_length);
    bit_position += flag_bit_length;
    sig->buf_len = (bit_position + 7) / 8;

    // Write header byte
    sig->buf[0] = ((uint8_t)(sig->buf_len));
    if (ep != -1) sig->buf[0] |= 1<<5;
    if (castle) sig->buf[0] |= 1<<6;
}
static int32_t ctg_signature_to_hash(ctg_signature_t* sig)
{
    static const uint32_t hash_bits[64] = {
        0x3100d2bf, 0x3118e3de, 0x34ab1372, 0x2807a847,
        0x1633f566, 0x2143b359, 0x26d56488, 0x3b9e6f59,
        0x37755656, 0x3089ca7b, 0x18e92d85, 0x0cd0e9d8,
        0x1a9e3b54, 0x3eaa902f, 0x0d9bfaae, 0x2f32b45b,
        0x31ed6102, 0x3d3c8398, 0x146660e3, 0x0f8d4b76,
        0x02c77a5f, 0x146c8799, 0x1c47f51f, 0x249f8f36,
        0x24772043, 0x1fbc1e4d, 0x1e86b3fa, 0x37df36a6,
        0x16ed30e4, 0x02c3148e, 0x216e5929, 0x0636b34e,
        0x317f9f56, 0x15f09d70, 0x131026fb, 0x38c784b1,
        0x29ac3305, 0x2b485dc5, 0x3c049ddc, 0x35a9fbcd,
        0x31d5373b, 0x2b246799, 0x0a2923d3, 0x08a96e9d,
        0x30031a9f, 0x08f525b5, 0x33611c06, 0x2409db98,
        0x0ca4feb2, 0x1000b71e, 0x30566e32, 0x39447d31,
        0x194e3752, 0x08233a95, 0x0f38fe36, 0x29c7cd57,
        0x0f7b3a39, 0x328e8a16, 0x1e7d1388, 0x0fba78f5,
        0x274c7e7c, 0x1e8be65c, 0x2fa0b0bb, 0x1eb6c371
    };

    int32_t hash = 0;
    int16_t tmp = 0;
    for (int i=0; i<sig->buf_len; ++i) {
        int8_t byte = sig->buf[i];
        tmp += ((0x0f - (byte & 0x0f)) << 2) + 1;
        hash += hash_bits[tmp & 0x3f];
        tmp += ((0xf0 - (byte & 0xf0)) >> 2) + 1;
        hash += hash_bits[tmp & 0x3f];
    }
    return hash;
}
static bool ctg_get_page_index(int hash, int* page_index)
{
    uint32_t key = 0;
    for (int mask = 1; key <= (uint32_t)page_bounds.high;
            mask = (mask << 1) + 1) {
        key = (hash & mask) + mask;
        if (key >= (uint32_t)page_bounds.low) {
            //printf("found entry with key=%d\n", key);
            fseek(cto_file, 16 + key*4, SEEK_SET);
            fread(page_index, 4, 1, cto_file);
            *page_index = my_ntohl((uint32_t)*page_index);
            if (*page_index >= 0) return true;
        }
    }
    //printf("didn't find entry\n");
    return false;
}
static bool ctg_lookup_entry(int page_index,
        ctg_signature_t* sig,
        ctg_entry_t* entry)
{
    // Pages are a uniform 4096 bytes.
    uint8_t buf[4096];
    fseek(ctg_file, 4096*(page_index + 1), SEEK_SET);
    if (!fread(buf, 1, 4096, ctg_file)) return false;
    int num_positions = (buf[0]<<8) + buf[1];
    //printf("found %d positions\n", num_positions);

    // Just scan through the list until we find a matching signature.
    int pos = 4;
    for (int i=0; i<num_positions; ++i) {
        int entry_size = buf[pos] % 32;
        bool equal = true;
        if (sig->buf_len != entry_size) equal = false;
        for (int j=0; j<sig->buf_len && equal; ++j) {
            if (buf[pos+j] != sig->buf[j]) equal = false;
        }
        if (!equal) {
            pos += entry_size + buf[pos+entry_size] + 33;
            continue;
        }
#define read_24(buf, pos)   \
    ((buf[pos]<<16) + (buf[(pos)+1]<<8) + (buf[(pos)+2]))
#define read_32(buf, pos)   \
    ((buf[pos]<<24) + (buf[pos+1]<<16) + (buf[(pos)+2]<<8) + (buf[(pos+3)+2]))
        // Found it, fill in the entry and return. Annoyingly, most of the
        // fields are 24 bits long.
        pos += entry_size;
        entry_size = buf[pos];
        for (int j=1; j<entry_size; ++j) entry->moves[j-1] = buf[pos+j];
        entry->num_moves = (entry_size - 1)/2;
        pos += entry_size;
        entry->total = read_24(buf, pos);
        pos += 3;
        entry->losses = read_24(buf, pos);
        pos += 3;
        entry->wins = read_24(buf, pos);
        pos += 3;
        entry->draws = read_24(buf, pos);
        pos += 3;
        entry->unknown1 = read_32(buf, pos);
        pos += 4;
        entry->avg_rating_games = read_24(buf, pos);
        pos += 3;
        entry->avg_rating_score = read_32(buf, pos);
        pos += 4;
        entry->perf_rating_games = read_24(buf, pos);
        pos += 3;
        entry->perf_rating_score = read_32(buf, pos);
        pos += 4;
        entry->recommendation = buf[pos];
        pos += 1;
        entry->unknown2 = buf[pos];
        pos += 1;
        entry->comment = buf[pos];
        return true;
    }
    return false;
}
static bool ctg_get_entry(position_t* pos, ctg_entry_t* entry)
{
    ctg_signature_t sig;
    position_to_ctg_signature(pos, &sig);
    int page_index, hash = ctg_signature_to_hash(&sig);
    if (!ctg_get_page_index(hash, &page_index)) return false;
    if (!ctg_lookup_entry(page_index, &sig, entry)) return false;
    return true;
}
static move_t squares_to_move(position_t* pos, square_t from, square_t to)
{
    move_t possible_moves[256];
    int num_moves = generate_legal_moves(pos, possible_moves);
    move_t move;
    for (int i=0; i<num_moves; ++i) {
        move = possible_moves[i];
        if (from == get_move_from(move) &&
                to == get_move_to(move) &&
                (get_move_promote(move) == NONE ||
                 get_move_promote(move) == QUEEN)) return move;
    }
    assert(false);
    return NO_MOVE;
}
static move_t byte_to_move(position_t* pos, uint8_t byte)
{
    const char* piece_code =
        "PNxQPQPxQBKxPBRNxxBKPBxxPxQBxBxxxRBQPxBPQQNxxPBQNQBxNxNQQQBQBxxx"
        "xQQxKQxxxxPQNQxxRxRxBPxxxxxxPxxPxQPQxxBKxRBxxxRQxxBxQxxxxBRRPRQR"
        "QRPxxNRRxxNPKxQQxxQxQxPKRRQPxQxBQxQPxRxxxRxQxRQxQPBxxRxQxBxPQQKx"
        "xBBBRRQPPQBPBRxPxPNNxxxQRQNPxxPKNRxRxQPQRNxPPQQRQQxNRBxNQQQQxQQx";
    const int piece_index[256]= {
        5, 2, 9, 2, 2, 1, 4, 9, 2, 2, 1, 9, 1, 1, 2, 1,
        9, 9, 1, 1, 8, 1, 9, 9, 7, 9, 2, 1, 9, 2, 9, 9,
        9, 2, 2, 2, 8, 9, 1, 3, 1, 1, 2, 9, 9, 6, 1, 1,
        2, 1, 2, 9, 1, 9, 1, 1, 2, 1, 1, 2, 1, 9, 9, 9,
        9, 2, 1, 9, 1, 1, 9, 9, 9, 9, 8, 1, 2, 2, 9, 9,
        1, 9, 1, 9, 2, 3, 9, 9, 9, 9, 9, 9, 7, 9, 9, 5,
        9, 1, 2, 2, 9, 9, 1, 1, 9, 2, 1, 0, 9, 9, 1, 2,
        9, 9, 2, 9, 1, 9, 9, 9, 9, 2, 1, 2, 3, 2, 1, 1,
        1, 1, 6, 9, 9, 1, 1, 1, 9, 9, 1, 1, 1, 9, 2, 1,
        9, 9, 2, 9, 1, 9, 2, 1, 1, 1, 1, 3, 9, 1, 9, 2,
        2, 9, 1, 8, 9, 2, 9, 9, 9, 2, 9, 2, 9, 2, 2, 9,
        2, 6, 1, 9, 9, 2, 9, 1, 9, 2, 9, 5, 2, 2, 1, 9,
        9, 1, 2, 1, 2, 2, 2, 7, 7, 2, 2, 6, 2, 1, 9, 4,
        9, 2, 2, 2, 9, 9, 9, 1, 2, 1, 1, 1, 9, 9, 5, 1,
        2, 1, 9, 2, 9, 1, 4, 1, 1, 1, 9, 4, 1, 1, 2, 1,
        2, 1, 9, 2, 2, 2, 0, 1, 2, 2, 2, 2, 9, 1, 2, 9
    };
    const int forward[256]= {
        1,-1, 9, 0, 1, 1, 1, 9, 0, 6,-1, 9, 1, 3, 0,-1,
        9, 9, 7, 1, 1, 5, 9, 9, 1, 9, 6, 1, 9, 7, 9, 9,
        9, 0, 2, 6, 1, 9, 7, 1, 5, 0,-2, 9, 9, 1, 1, 0,
       -2, 0, 5, 9, 2, 9, 1, 4, 4, 0, 6, 5, 5, 9, 9, 9,
        9, 5, 7, 9,-1, 3, 9, 9, 9, 9, 2, 5, 2, 1, 9, 9,
        6, 9, 0, 9, 1, 1, 9, 9, 9, 9, 9, 9, 1, 9, 9, 2,
        9, 6, 2, 7, 9, 9, 3, 1, 9, 7, 4, 0, 9, 9, 0, 7,
        9, 9, 7, 9, 0, 9, 9, 9, 9, 6, 3, 6, 1, 1, 3, 0,
        6, 1, 1, 9, 9, 2, 0, 5, 9, 9,-2, 1,-1, 9, 2, 0,
        9, 9, 1, 9, 3, 9, 1, 0, 0, 4, 6, 2, 9, 2, 9, 4,
        3, 9, 2, 1, 9, 5, 9, 9, 9, 0, 9, 6, 9, 0, 3, 9,
        4, 2, 6, 9, 9, 0, 9, 5, 9, 3, 9, 1, 0, 2, 0, 9,
        9, 2, 2, 2, 0, 4, 5, 1, 2, 7, 3, 1, 5, 0, 9, 1,
        9, 1, 1, 1, 9, 9, 9, 1, 0, 2,-2, 2, 9, 9, 1, 1,
       -1, 7, 9, 3, 9, 0, 2, 4, 2,-1, 9, 1, 1, 7, 1, 0,
        0, 1, 9, 2, 2, 1, 0, 1, 0, 6, 0, 2, 9, 7, 3, 9
    };
    const int left[256] = {
       -1, 2, 9,-2, 0, 0, 1, 9,-4,-6, 0, 9, 1,-3,-3, 2,
        9, 9,-7, 0,-1,-5, 9, 9, 0, 9, 0, 1, 9,-7, 9, 9,
        9,-7, 2,-6, 1, 9, 7, 1,-5,-6,-1, 9, 9,-1,-1,-1,
        1,-3,-5, 9,-1, 9,-2, 0, 4,-5,-6, 5, 5, 9, 9, 9,
        9,-5, 7, 9,-1,-3, 9, 9, 9, 9, 0, 5,-1, 0, 9, 9,
        0, 9,-6, 9, 1, 0, 9, 9, 9, 9, 9, 9,-1, 9, 9, 0,
        9,-6, 0, 7, 9, 9, 3,-1, 9, 0,-4, 0, 9, 9,-5,-7,
        9, 9, 7, 9,-2, 9, 9, 9, 9, 6, 0, 0,-1, 0, 3,-1,
        6, 0, 1, 9, 9, 1,-7, 0, 9, 9,-1,-1, 1, 9, 2,-7,
        9, 9,-1, 9, 0, 9,-1, 1,-3, 0, 0, 0, 9, 0, 9, 4,
        0, 9,-2, 0, 9, 0, 9, 9, 9,-2, 9, 6, 9,-4,-3, 9,
        0, 0, 6, 9, 9,-5, 9, 0, 9,-3, 9, 0,-5, 0,-1, 9,
        9,-2,-2, 2,-1, 0, 0, 1, 0, 0, 3, 0, 5,-2, 9, 0,
        9, 1,-2, 2, 9, 9, 9, 1,-6, 2, 1, 0, 9, 9, 1, 1,
       -2, 0, 9, 0, 9,-4, 0,-4, 0,-2, 9,-1, 0,-7, 1,-4,
       -7,-1, 9, 1, 0,-1, 0, 2,-1, 0,-3,-2, 9, 0, 3, 9
    };

    // Find the piece. Note: the board may be mirrored/flipped.
    bool flip_board = pos->side_to_move == BLACK;
    color_t white = flip_board ? BLACK : WHITE;
    bool mirror_board = square_file(pos->pieces[white][0]) < FILE_E &&
        pos->castle_rights == 0;
    int file_from = -1, file_to = -1, rank_from = -1, rank_to = -1;

    // Handle castling.
    if (byte == 107) {
        file_from = 4;
        file_to = 6;
        rank_from = rank_to = flip_board ? 7 : 0;
        return squares_to_move(pos,
                create_square(file_from, rank_from),
                create_square(file_to, rank_to));
    }
    if (byte == 246) {
        file_from = 4;
        file_to = 2;
        rank_from = rank_to = flip_board ? 7 : 0;
        return squares_to_move(pos,
                create_square(file_from, rank_from),
                create_square(file_to, rank_to));
    }

    // Look up piece type. Note: positions are always white to move.
    piece_t pc = NONE;
    char glyph = piece_code[byte];
    switch (piece_code[byte]) {
        case 'P': pc = WP; break;
        case 'N': pc = WN; break;
        case 'B': pc = WB; break;
        case 'R': pc = WR; break;
        case 'Q': pc = WQ; break;
        case 'K': pc = WK; break;
        default: printf("%d -> (%c)\n", byte, glyph); assert(false);
    }

    // Find the piece.
    int nth_piece = piece_index[byte], piece_count = 0;
    bool found = false;
    for (int file=0; file<8 && !found; ++file) {
        for (int rank=0; rank<8 && !found; ++rank) {
            square_t sq = create_square(file, rank);
            if (flip_board) sq = mirror_rank(sq);
            if (mirror_board) sq = mirror_file(sq);
            piece_t piece = flip_board ?
                flip_piece[pos->board[sq]] : pos->board[sq];
            if (piece == pc) piece_count++;
            if (piece_count == nth_piece) {
                file_from = file;
                rank_from = rank;
                found = true;
            }
        }
    }
    assert(found);

    // Normalize rank and file values.
    file_to = file_from - left[byte];
    file_to = (file_to + 8) % 8;
    rank_to = rank_from + forward[byte];
    rank_to = (rank_to + 8) % 8;
    if (flip_board) {
        rank_from = 7-rank_from;
        rank_to = 7-rank_to;
    }
    if (mirror_board) {
        file_from = 7-file_from;
        file_to = 7-file_to;
    }
    return squares_to_move(pos,
            create_square(file_from, rank_from),
            create_square(file_to, rank_to));
}
void move_to_coord_str(move_t move, char* str)
{
    if (move == NO_MOVE) {
        strcpy(str, "(none)");
        return;
    }
    if (move == NULL_MOVE) {
        strcpy(str, "0000");
        return;
    }
    square_t from = get_move_from(move);
    square_t to = get_move_to(move);
    if (options.chess960) {
        if (is_move_castle_long(move)) {
            if (options.arena_castle) {
                if (queen_rook_home != A1 || king_home != E1) {
                    strcpy(str, "O-O-O");
                    return;
                }
            } else to = queen_rook_home + get_move_piece_color(move)*A8;
        } else if (is_move_castle_short(move)) {
            if (options.arena_castle) {
                if (king_rook_home != H1 || king_home != E1) {
                    strcpy(str, "O-O");
                    return;
                }
            } else to = king_rook_home + get_move_piece_color(move)*A8;
        }
    }
    str += snprintf(str, 5, "%c%c%c%c",
           (char)square_file(from) + 'a', (char)square_rank(from) + '1',
           (char)square_file(to) + 'a', (char)square_rank(to) + '1');
    if (get_move_promote(move)) {
        snprintf(str, 2, "%c", tolower(glyphs[get_move_promote(move)]));
    }
}

void print_coord_move(move_t move)
{
    static char move_str[7];
    move_to_coord_str(move, move_str);
    printf("%s ", move_str);
}
static int64_t move_weight(position_t* pos,
        move_t move,
        uint8_t annotation,
        bool* recommended)
{
    undo_info_t undo;
    do_move(pos, move, &undo);
    ctg_entry_t entry;
    bool success = ctg_get_entry(pos, &entry);
    undo_move(pos, move, &undo);
    if (!success) return 0;

    *recommended = false;
    int64_t half_points = 2*entry.wins + entry.draws;
    int64_t games = entry.wins + entry.draws + entry.losses;
    int64_t weight = (games < 1) ? 0 : (half_points * 10000) / games;
    if (entry.recommendation == 64) weight = 0;
    if (entry.recommendation == 128) *recommended = true;

    // Adjust weights based on move annotations. Note that moves can be both
    // marked as recommended and annotated with a '?'. Since moves like this
    // are not marked green in GUI tools, the recommendation is turned off in
    // order to give results consistent with expectations.
    switch (annotation) {
        case 0x01: weight *=  8; break;                         //  !
        case 0x02: weight  =  0; *recommended = false; break;   //  ?
        case 0x03: weight *= 32; break;                         // !!
        case 0x04: weight  =  0; *recommended = false; break;   // ??
        case 0x05: weight /=  2; *recommended = false; break;   // !?
        case 0x06: weight /=  8; *recommended = false; break;   // ?!
        case 0x08: weight = INT32_MAX; break;                   // Only move
        case 0x16: break;                                       // Zugzwang
        default: break;
    }
    printf("info string book move ");
    print_coord_move(move);
    //printf("weight %6"PRIu64"\n", weight);
    //printf("weight %6"PRIu64" wins %6d draws %6d losses %6d rec %3d "
    //        "note %2d avg_games %6d avg_score %9d "
    //        "perf_games %6d perf_score %9d\n",
    //        weight, entry.wins, entry.draws, entry.losses, entry.recommendation,
    //        annotation, entry.avg_rating_games, entry.avg_rating_score,
    //        entry.perf_rating_games, entry.perf_rating_score);
    return weight;
}
int64_t random_64(void)
{
    int64_t r = random_32();
    r <<= 32;
    r |= random_32();
    return r;
}
static bool ctg_pick_move(position_t* pos, ctg_entry_t* entry, move_t* move)
{
    move_t moves[50];
    int64_t weights[50];
    bool recommended[50];
    int64_t total_weight = 0;
    bool have_recommendations = false;
    for (int i=0; i<2*entry->num_moves; i += 2) {
        uint8_t byte = entry->moves[i];
        move_t m = byte_to_move(pos, byte);
        moves[i/2] = m;
        weights[i/2] = move_weight(pos, m, entry->moves[i+1], &recommended[i/2]);
        if (recommended[i/2]) have_recommendations = true;
        if (move == NO_MOVE) break;
    }

    // Do a prefix sum on the weights to facilitate a random choice. If there are recommended
    // moves, ensure that we don't pick a move that wasn't recommended.
    for (int i=0; i<entry->num_moves; ++i) {
        if (have_recommendations && !recommended[i]) weights[i] = 0;
        total_weight += weights[i];
        weights[i] = total_weight;
    }
    if (total_weight == 0) {
        *move = NO_MOVE;
        return false;
    }
    int64_t choice = random_64() % total_weight;
    int64_t i;
    for (i=0; choice >= weights[i]; ++i) {}
    if (i >= entry->num_moves) {
        //printf("i: %"PRIu64"\nchoice: %"PRIu64"\ntotal_weight: %"
                //PRIu64"\nnum_moves: %d\n",
                //i, choice, total_weight, entry->num_moves);
        assert(false);
    }
    *move = moves[i];
    return true;
}
move_t get_ctg_book_move(position_t* pos)
{
    move_t move;
    ctg_entry_t entry;
    if (!ctg_get_entry(pos, &entry)) return NO_MOVE;
    if (!ctg_pick_move(pos, &entry, &move)) return NO_MOVE;
    return move;
}
typedef struct {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
} book_entry_t;

static FILE* book = NULL;
static int num_entries;
bool init_poly_book(char* filename)
{
    assert(sizeof(book_entry_t) == 16);
    srandom_32(time(NULL));
    if (book) fclose(book);
    if (!(book = fopen(filename, "r"))) {
        num_entries = 0;
        return false;
    }
    fseek(book, 0, SEEK_END);
    num_entries = ftell(book) / 16;
    return true;
}
const int book_piece_index[16] = {
    0, 1, 3, 5, 7, 9, 11, 0,
    0, 0, 2, 4, 6, 8, 10
};
const uint64_t book_random[781] = {
    0x9D39247E33776D41ULL, 0x2AF7398005AAA5C7ULL, 0x44DB015024623547ULL,
    0x9C15F73E62A76AE2ULL, 0x75834465489C0C89ULL, 0x3290AC3A203001BFULL,
    0x0FBBAD1F61042279ULL, 0xE83A908FF2FB60CAULL, 0x0D7E765D58755C10ULL,
    0x1A083822CEAFE02DULL, 0x9605D5F0E25EC3B0ULL, 0xD021FF5CD13A2ED5ULL,
    0x40BDF15D4A672E32ULL, 0x011355146FD56395ULL, 0x5DB4832046F3D9E5ULL,
    0x239F8B2D7FF719CCULL, 0x05D1A1AE85B49AA1ULL, 0x679F848F6E8FC971ULL,
    0x7449BBFF801FED0BULL, 0x7D11CDB1C3B7ADF0ULL, 0x82C7709E781EB7CCULL,
    0xF3218F1C9510786CULL, 0x331478F3AF51BBE6ULL, 0x4BB38DE5E7219443ULL,
    0xAA649C6EBCFD50FCULL, 0x8DBD98A352AFD40BULL, 0x87D2074B81D79217ULL,
    0x19F3C751D3E92AE1ULL, 0xB4AB30F062B19ABFULL, 0x7B0500AC42047AC4ULL,
    0xC9452CA81A09D85DULL, 0x24AA6C514DA27500ULL, 0x4C9F34427501B447ULL,
    0x14A68FD73C910841ULL, 0xA71B9B83461CBD93ULL, 0x03488B95B0F1850FULL,
    0x637B2B34FF93C040ULL, 0x09D1BC9A3DD90A94ULL, 0x3575668334A1DD3BULL,
    0x735E2B97A4C45A23ULL, 0x18727070F1BD400BULL, 0x1FCBACD259BF02E7ULL,
    0xD310A7C2CE9B6555ULL, 0xBF983FE0FE5D8244ULL, 0x9F74D14F7454A824ULL,
    0x51EBDC4AB9BA3035ULL, 0x5C82C505DB9AB0FAULL, 0xFCF7FE8A3430B241ULL,
    0x3253A729B9BA3DDEULL, 0x8C74C368081B3075ULL, 0xB9BC6C87167C33E7ULL,
    0x7EF48F2B83024E20ULL, 0x11D505D4C351BD7FULL, 0x6568FCA92C76A243ULL,
    0x4DE0B0F40F32A7B8ULL, 0x96D693460CC37E5DULL, 0x42E240CB63689F2FULL,
    0x6D2BDCDAE2919661ULL, 0x42880B0236E4D951ULL, 0x5F0F4A5898171BB6ULL,
    0x39F890F579F92F88ULL, 0x93C5B5F47356388BULL, 0x63DC359D8D231B78ULL,
    0xEC16CA8AEA98AD76ULL, 0x5355F900C2A82DC7ULL, 0x07FB9F855A997142ULL,
    0x5093417AA8A7ED5EULL, 0x7BCBC38DA25A7F3CULL, 0x19FC8A768CF4B6D4ULL,
    0x637A7780DECFC0D9ULL, 0x8249A47AEE0E41F7ULL, 0x79AD695501E7D1E8ULL,
    0x14ACBAF4777D5776ULL, 0xF145B6BECCDEA195ULL, 0xDABF2AC8201752FCULL,
    0x24C3C94DF9C8D3F6ULL, 0xBB6E2924F03912EAULL, 0x0CE26C0B95C980D9ULL,
    0xA49CD132BFBF7CC4ULL, 0xE99D662AF4243939ULL, 0x27E6AD7891165C3FULL,
    0x8535F040B9744FF1ULL, 0x54B3F4FA5F40D873ULL, 0x72B12C32127FED2BULL,
    0xEE954D3C7B411F47ULL, 0x9A85AC909A24EAA1ULL, 0x70AC4CD9F04F21F5ULL,
    0xF9B89D3E99A075C2ULL, 0x87B3E2B2B5C907B1ULL, 0xA366E5B8C54F48B8ULL,
    0xAE4A9346CC3F7CF2ULL, 0x1920C04D47267BBDULL, 0x87BF02C6B49E2AE9ULL,
    0x092237AC237F3859ULL, 0xFF07F64EF8ED14D0ULL, 0x8DE8DCA9F03CC54EULL,
    0x9C1633264DB49C89ULL, 0xB3F22C3D0B0B38EDULL, 0x390E5FB44D01144BULL,
    0x5BFEA5B4712768E9ULL, 0x1E1032911FA78984ULL, 0x9A74ACB964E78CB3ULL,
    0x4F80F7A035DAFB04ULL, 0x6304D09A0B3738C4ULL, 0x2171E64683023A08ULL,
    0x5B9B63EB9CEFF80CULL, 0x506AACF489889342ULL, 0x1881AFC9A3A701D6ULL,
    0x6503080440750644ULL, 0xDFD395339CDBF4A7ULL, 0xEF927DBCF00C20F2ULL,
    0x7B32F7D1E03680ECULL, 0xB9FD7620E7316243ULL, 0x05A7E8A57DB91B77ULL,
    0xB5889C6E15630A75ULL, 0x4A750A09CE9573F7ULL, 0xCF464CEC899A2F8AULL,
    0xF538639CE705B824ULL, 0x3C79A0FF5580EF7FULL, 0xEDE6C87F8477609DULL,
    0x799E81F05BC93F31ULL, 0x86536B8CF3428A8CULL, 0x97D7374C60087B73ULL,
    0xA246637CFF328532ULL, 0x043FCAE60CC0EBA0ULL, 0x920E449535DD359EULL,
    0x70EB093B15B290CCULL, 0x73A1921916591CBDULL, 0x56436C9FE1A1AA8DULL,
    0xEFAC4B70633B8F81ULL, 0xBB215798D45DF7AFULL, 0x45F20042F24F1768ULL,
    0x930F80F4E8EB7462ULL, 0xFF6712FFCFD75EA1ULL, 0xAE623FD67468AA70ULL,
    0xDD2C5BC84BC8D8FCULL, 0x7EED120D54CF2DD9ULL, 0x22FE545401165F1CULL,
    0xC91800E98FB99929ULL, 0x808BD68E6AC10365ULL, 0xDEC468145B7605F6ULL,
    0x1BEDE3A3AEF53302ULL, 0x43539603D6C55602ULL, 0xAA969B5C691CCB7AULL,
    0xA87832D392EFEE56ULL, 0x65942C7B3C7E11AEULL, 0xDED2D633CAD004F6ULL,
    0x21F08570F420E565ULL, 0xB415938D7DA94E3CULL, 0x91B859E59ECB6350ULL,
    0x10CFF333E0ED804AULL, 0x28AED140BE0BB7DDULL, 0xC5CC1D89724FA456ULL,
    0x5648F680F11A2741ULL, 0x2D255069F0B7DAB3ULL, 0x9BC5A38EF729ABD4ULL,
    0xEF2F054308F6A2BCULL, 0xAF2042F5CC5C2858ULL, 0x480412BAB7F5BE2AULL,
    0xAEF3AF4A563DFE43ULL, 0x19AFE59AE451497FULL, 0x52593803DFF1E840ULL,
    0xF4F076E65F2CE6F0ULL, 0x11379625747D5AF3ULL, 0xBCE5D2248682C115ULL,
    0x9DA4243DE836994FULL, 0x066F70B33FE09017ULL, 0x4DC4DE189B671A1CULL,
    0x51039AB7712457C3ULL, 0xC07A3F80C31FB4B4ULL, 0xB46EE9C5E64A6E7CULL,
    0xB3819A42ABE61C87ULL, 0x21A007933A522A20ULL, 0x2DF16F761598AA4FULL,
    0x763C4A1371B368FDULL, 0xF793C46702E086A0ULL, 0xD7288E012AEB8D31ULL,
    0xDE336A2A4BC1C44BULL, 0x0BF692B38D079F23ULL, 0x2C604A7A177326B3ULL,
    0x4850E73E03EB6064ULL, 0xCFC447F1E53C8E1BULL, 0xB05CA3F564268D99ULL,
    0x9AE182C8BC9474E8ULL, 0xA4FC4BD4FC5558CAULL, 0xE755178D58FC4E76ULL,
    0x69B97DB1A4C03DFEULL, 0xF9B5B7C4ACC67C96ULL, 0xFC6A82D64B8655FBULL,
    0x9C684CB6C4D24417ULL, 0x8EC97D2917456ED0ULL, 0x6703DF9D2924E97EULL,
    0xC547F57E42A7444EULL, 0x78E37644E7CAD29EULL, 0xFE9A44E9362F05FAULL,
    0x08BD35CC38336615ULL, 0x9315E5EB3A129ACEULL, 0x94061B871E04DF75ULL,
    0xDF1D9F9D784BA010ULL, 0x3BBA57B68871B59DULL, 0xD2B7ADEEDED1F73FULL,
    0xF7A255D83BC373F8ULL, 0xD7F4F2448C0CEB81ULL, 0xD95BE88CD210FFA7ULL,
    0x336F52F8FF4728E7ULL, 0xA74049DAC312AC71ULL, 0xA2F61BB6E437FDB5ULL,
    0x4F2A5CB07F6A35B3ULL, 0x87D380BDA5BF7859ULL, 0x16B9F7E06C453A21ULL,
    0x7BA2484C8A0FD54EULL, 0xF3A678CAD9A2E38CULL, 0x39B0BF7DDE437BA2ULL,
    0xFCAF55C1BF8A4424ULL, 0x18FCF680573FA594ULL, 0x4C0563B89F495AC3ULL,
    0x40E087931A00930DULL, 0x8CFFA9412EB642C1ULL, 0x68CA39053261169FULL,
    0x7A1EE967D27579E2ULL, 0x9D1D60E5076F5B6FULL, 0x3810E399B6F65BA2ULL,
    0x32095B6D4AB5F9B1ULL, 0x35CAB62109DD038AULL, 0xA90B24499FCFAFB1ULL,
    0x77A225A07CC2C6BDULL, 0x513E5E634C70E331ULL, 0x4361C0CA3F692F12ULL,
    0xD941ACA44B20A45BULL, 0x528F7C8602C5807BULL, 0x52AB92BEB9613989ULL,
    0x9D1DFA2EFC557F73ULL, 0x722FF175F572C348ULL, 0x1D1260A51107FE97ULL,
    0x7A249A57EC0C9BA2ULL, 0x04208FE9E8F7F2D6ULL, 0x5A110C6058B920A0ULL,
    0x0CD9A497658A5698ULL, 0x56FD23C8F9715A4CULL, 0x284C847B9D887AAEULL,
    0x04FEABFBBDB619CBULL, 0x742E1E651C60BA83ULL, 0x9A9632E65904AD3CULL,
    0x881B82A13B51B9E2ULL, 0x506E6744CD974924ULL, 0xB0183DB56FFC6A79ULL,
    0x0ED9B915C66ED37EULL, 0x5E11E86D5873D484ULL, 0xF678647E3519AC6EULL,
    0x1B85D488D0F20CC5ULL, 0xDAB9FE6525D89021ULL, 0x0D151D86ADB73615ULL,
    0xA865A54EDCC0F019ULL, 0x93C42566AEF98FFBULL, 0x99E7AFEABE000731ULL,
    0x48CBFF086DDF285AULL, 0x7F9B6AF1EBF78BAFULL, 0x58627E1A149BBA21ULL,
    0x2CD16E2ABD791E33ULL, 0xD363EFF5F0977996ULL, 0x0CE2A38C344A6EEDULL,
    0x1A804AADB9CFA741ULL, 0x907F30421D78C5DEULL, 0x501F65EDB3034D07ULL,
    0x37624AE5A48FA6E9ULL, 0x957BAF61700CFF4EULL, 0x3A6C27934E31188AULL,
    0xD49503536ABCA345ULL, 0x088E049589C432E0ULL, 0xF943AEE7FEBF21B8ULL,
    0x6C3B8E3E336139D3ULL, 0x364F6FFA464EE52EULL, 0xD60F6DCEDC314222ULL,
    0x56963B0DCA418FC0ULL, 0x16F50EDF91E513AFULL, 0xEF1955914B609F93ULL,
    0x565601C0364E3228ULL, 0xECB53939887E8175ULL, 0xBAC7A9A18531294BULL,
    0xB344C470397BBA52ULL, 0x65D34954DAF3CEBDULL, 0xB4B81B3FA97511E2ULL,
    0xB422061193D6F6A7ULL, 0x071582401C38434DULL, 0x7A13F18BBEDC4FF5ULL,
    0xBC4097B116C524D2ULL, 0x59B97885E2F2EA28ULL, 0x99170A5DC3115544ULL,
    0x6F423357E7C6A9F9ULL, 0x325928EE6E6F8794ULL, 0xD0E4366228B03343ULL,
    0x565C31F7DE89EA27ULL, 0x30F5611484119414ULL, 0xD873DB391292ED4FULL,
    0x7BD94E1D8E17DEBCULL, 0xC7D9F16864A76E94ULL, 0x947AE053EE56E63CULL,
    0xC8C93882F9475F5FULL, 0x3A9BF55BA91F81CAULL, 0xD9A11FBB3D9808E4ULL,
    0x0FD22063EDC29FCAULL, 0xB3F256D8ACA0B0B9ULL, 0xB03031A8B4516E84ULL,
    0x35DD37D5871448AFULL, 0xE9F6082B05542E4EULL, 0xEBFAFA33D7254B59ULL,
    0x9255ABB50D532280ULL, 0xB9AB4CE57F2D34F3ULL, 0x693501D628297551ULL,
    0xC62C58F97DD949BFULL, 0xCD454F8F19C5126AULL, 0xBBE83F4ECC2BDECBULL,
    0xDC842B7E2819E230ULL, 0xBA89142E007503B8ULL, 0xA3BC941D0A5061CBULL,
    0xE9F6760E32CD8021ULL, 0x09C7E552BC76492FULL, 0x852F54934DA55CC9ULL,
    0x8107FCCF064FCF56ULL, 0x098954D51FFF6580ULL, 0x23B70EDB1955C4BFULL,
    0xC330DE426430F69DULL, 0x4715ED43E8A45C0AULL, 0xA8D7E4DAB780A08DULL,
    0x0572B974F03CE0BBULL, 0xB57D2E985E1419C7ULL, 0xE8D9ECBE2CF3D73FULL,
    0x2FE4B17170E59750ULL, 0x11317BA87905E790ULL, 0x7FBF21EC8A1F45ECULL,
    0x1725CABFCB045B00ULL, 0x964E915CD5E2B207ULL, 0x3E2B8BCBF016D66DULL,
    0xBE7444E39328A0ACULL, 0xF85B2B4FBCDE44B7ULL, 0x49353FEA39BA63B1ULL,
    0x1DD01AAFCD53486AULL, 0x1FCA8A92FD719F85ULL, 0xFC7C95D827357AFAULL,
    0x18A6A990C8B35EBDULL, 0xCCCB7005C6B9C28DULL, 0x3BDBB92C43B17F26ULL,
    0xAA70B5B4F89695A2ULL, 0xE94C39A54A98307FULL, 0xB7A0B174CFF6F36EULL,
    0xD4DBA84729AF48ADULL, 0x2E18BC1AD9704A68ULL, 0x2DE0966DAF2F8B1CULL,
    0xB9C11D5B1E43A07EULL, 0x64972D68DEE33360ULL, 0x94628D38D0C20584ULL,
    0xDBC0D2B6AB90A559ULL, 0xD2733C4335C6A72FULL, 0x7E75D99D94A70F4DULL,
    0x6CED1983376FA72BULL, 0x97FCAACBF030BC24ULL, 0x7B77497B32503B12ULL,
    0x8547EDDFB81CCB94ULL, 0x79999CDFF70902CBULL, 0xCFFE1939438E9B24ULL,
    0x829626E3892D95D7ULL, 0x92FAE24291F2B3F1ULL, 0x63E22C147B9C3403ULL,
    0xC678B6D860284A1CULL, 0x5873888850659AE7ULL, 0x0981DCD296A8736DULL,
    0x9F65789A6509A440ULL, 0x9FF38FED72E9052FULL, 0xE479EE5B9930578CULL,
    0xE7F28ECD2D49EECDULL, 0x56C074A581EA17FEULL, 0x5544F7D774B14AEFULL,
    0x7B3F0195FC6F290FULL, 0x12153635B2C0CF57ULL, 0x7F5126DBBA5E0CA7ULL,
    0x7A76956C3EAFB413ULL, 0x3D5774A11D31AB39ULL, 0x8A1B083821F40CB4ULL,
    0x7B4A38E32537DF62ULL, 0x950113646D1D6E03ULL, 0x4DA8979A0041E8A9ULL,
    0x3BC36E078F7515D7ULL, 0x5D0A12F27AD310D1ULL, 0x7F9D1A2E1EBE1327ULL,
    0xDA3A361B1C5157B1ULL, 0xDCDD7D20903D0C25ULL, 0x36833336D068F707ULL,
    0xCE68341F79893389ULL, 0xAB9090168DD05F34ULL, 0x43954B3252DC25E5ULL,
    0xB438C2B67F98E5E9ULL, 0x10DCD78E3851A492ULL, 0xDBC27AB5447822BFULL,
    0x9B3CDB65F82CA382ULL, 0xB67B7896167B4C84ULL, 0xBFCED1B0048EAC50ULL,
    0xA9119B60369FFEBDULL, 0x1FFF7AC80904BF45ULL, 0xAC12FB171817EEE7ULL,
    0xAF08DA9177DDA93DULL, 0x1B0CAB936E65C744ULL, 0xB559EB1D04E5E932ULL,
    0xC37B45B3F8D6F2BAULL, 0xC3A9DC228CAAC9E9ULL, 0xF3B8B6675A6507FFULL,
    0x9FC477DE4ED681DAULL, 0x67378D8ECCEF96CBULL, 0x6DD856D94D259236ULL,
    0xA319CE15B0B4DB31ULL, 0x073973751F12DD5EULL, 0x8A8E849EB32781A5ULL,
    0xE1925C71285279F5ULL, 0x74C04BF1790C0EFEULL, 0x4DDA48153C94938AULL,
    0x9D266D6A1CC0542CULL, 0x7440FB816508C4FEULL, 0x13328503DF48229FULL,
    0xD6BF7BAEE43CAC40ULL, 0x4838D65F6EF6748FULL, 0x1E152328F3318DEAULL,
    0x8F8419A348F296BFULL, 0x72C8834A5957B511ULL, 0xD7A023A73260B45CULL,
    0x94EBC8ABCFB56DAEULL, 0x9FC10D0F989993E0ULL, 0xDE68A2355B93CAE6ULL,
    0xA44CFE79AE538BBEULL, 0x9D1D84FCCE371425ULL, 0x51D2B1AB2DDFB636ULL,
    0x2FD7E4B9E72CD38CULL, 0x65CA5B96B7552210ULL, 0xDD69A0D8AB3B546DULL,
    0x604D51B25FBF70E2ULL, 0x73AA8A564FB7AC9EULL, 0x1A8C1E992B941148ULL,
    0xAAC40A2703D9BEA0ULL, 0x764DBEAE7FA4F3A6ULL, 0x1E99B96E70A9BE8BULL,
    0x2C5E9DEB57EF4743ULL, 0x3A938FEE32D29981ULL, 0x26E6DB8FFDF5ADFEULL,
    0x469356C504EC9F9DULL, 0xC8763C5B08D1908CULL, 0x3F6C6AF859D80055ULL,
    0x7F7CC39420A3A545ULL, 0x9BFB227EBDF4C5CEULL, 0x89039D79D6FC5C5CULL,
    0x8FE88B57305E2AB6ULL, 0xA09E8C8C35AB96DEULL, 0xFA7E393983325753ULL,
    0xD6B6D0ECC617C699ULL, 0xDFEA21EA9E7557E3ULL, 0xB67C1FA481680AF8ULL,
    0xCA1E3785A9E724E5ULL, 0x1CFC8BED0D681639ULL, 0xD18D8549D140CAEAULL,
    0x4ED0FE7E9DC91335ULL, 0xE4DBF0634473F5D2ULL, 0x1761F93A44D5AEFEULL,
    0x53898E4C3910DA55ULL, 0x734DE8181F6EC39AULL, 0x2680B122BAA28D97ULL,
    0x298AF231C85BAFABULL, 0x7983EED3740847D5ULL, 0x66C1A2A1A60CD889ULL,
    0x9E17E49642A3E4C1ULL, 0xEDB454E7BADC0805ULL, 0x50B704CAB602C329ULL,
    0x4CC317FB9CDDD023ULL, 0x66B4835D9EAFEA22ULL, 0x219B97E26FFC81BDULL,
    0x261E4E4C0A333A9DULL, 0x1FE2CCA76517DB90ULL, 0xD7504DFA8816EDBBULL,
    0xB9571FA04DC089C8ULL, 0x1DDC0325259B27DEULL, 0xCF3F4688801EB9AAULL,
    0xF4F5D05C10CAB243ULL, 0x38B6525C21A42B0EULL, 0x36F60E2BA4FA6800ULL,
    0xEB3593803173E0CEULL, 0x9C4CD6257C5A3603ULL, 0xAF0C317D32ADAA8AULL,
    0x258E5A80C7204C4BULL, 0x8B889D624D44885DULL, 0xF4D14597E660F855ULL,
    0xD4347F66EC8941C3ULL, 0xE699ED85B0DFB40DULL, 0x2472F6207C2D0484ULL,
    0xC2A1E7B5B459AEB5ULL, 0xAB4F6451CC1D45ECULL, 0x63767572AE3D6174ULL,
    0xA59E0BD101731A28ULL, 0x116D0016CB948F09ULL, 0x2CF9C8CA052F6E9FULL,
    0x0B090A7560A968E3ULL, 0xABEEDDB2DDE06FF1ULL, 0x58EFC10B06A2068DULL,
    0xC6E57A78FBD986E0ULL, 0x2EAB8CA63CE802D7ULL, 0x14A195640116F336ULL,
    0x7C0828DD624EC390ULL, 0xD74BBE77E6116AC7ULL, 0x804456AF10F5FB53ULL,
    0xEBE9EA2ADF4321C7ULL, 0x03219A39EE587A30ULL, 0x49787FEF17AF9924ULL,
    0xA1E9300CD8520548ULL, 0x5B45E522E4B1B4EFULL, 0xB49C3B3995091A36ULL,
    0xD4490AD526F14431ULL, 0x12A8F216AF9418C2ULL, 0x001F837CC7350524ULL,
    0x1877B51E57A764D5ULL, 0xA2853B80F17F58EEULL, 0x993E1DE72D36D310ULL,
    0xB3598080CE64A656ULL, 0x252F59CF0D9F04BBULL, 0xD23C8E176D113600ULL,
    0x1BDA0492E7E4586EULL, 0x21E0BD5026C619BFULL, 0x3B097ADAF088F94EULL,
    0x8D14DEDB30BE846EULL, 0xF95CFFA23AF5F6F4ULL, 0x3871700761B3F743ULL,
    0xCA672B91E9E4FA16ULL, 0x64C8E531BFF53B55ULL, 0x241260ED4AD1E87DULL,
    0x106C09B972D2E822ULL, 0x7FBA195410E5CA30ULL, 0x7884D9BC6CB569D8ULL,
    0x0647DFEDCD894A29ULL, 0x63573FF03E224774ULL, 0x4FC8E9560F91B123ULL,
    0x1DB956E450275779ULL, 0xB8D91274B9E9D4FBULL, 0xA2EBEE47E2FBFCE1ULL,
    0xD9F1F30CCD97FB09ULL, 0xEFED53D75FD64E6BULL, 0x2E6D02C36017F67FULL,
    0xA9AA4D20DB084E9BULL, 0xB64BE8D8B25396C1ULL, 0x70CB6AF7C2D5BCF0ULL,
    0x98F076A4F7A2322EULL, 0xBF84470805E69B5FULL, 0x94C3251F06F90CF3ULL,
    0x3E003E616A6591E9ULL, 0xB925A6CD0421AFF3ULL, 0x61BDD1307C66E300ULL,
    0xBF8D5108E27E0D48ULL, 0x240AB57A8B888B20ULL, 0xFC87614BAF287E07ULL,
    0xEF02CDD06FFDB432ULL, 0xA1082C0466DF6C0AULL, 0x8215E577001332C8ULL,
    0xD39BB9C3A48DB6CFULL, 0x2738259634305C14ULL, 0x61CF4F94C97DF93DULL,
    0x1B6BACA2AE4E125BULL, 0x758F450C88572E0BULL, 0x959F587D507A8359ULL,
    0xB063E962E045F54DULL, 0x60E8ED72C0DFF5D1ULL, 0x7B64978555326F9FULL,
    0xFD080D236DA814BAULL, 0x8C90FD9B083F4558ULL, 0x106F72FE81E2C590ULL,
    0x7976033A39F7D952ULL, 0xA4EC0132764CA04BULL, 0x733EA705FAE4FA77ULL,
    0xB4D8F77BC3E56167ULL, 0x9E21F4F903B33FD9ULL, 0x9D765E419FB69F6DULL,
    0xD30C088BA61EA5EFULL, 0x5D94337FBFAF7F5BULL, 0x1A4E4822EB4D7A59ULL,
    0x6FFE73E81B637FB3ULL, 0xDDF957BC36D8B9CAULL, 0x64D0E29EEA8838B3ULL,
    0x08DD9BDFD96B9F63ULL, 0x087E79E5A57D1D13ULL, 0xE328E230E3E2B3FBULL,
    0x1C2559E30F0946BEULL, 0x720BF5F26F4D2EAAULL, 0xB0774D261CC609DBULL,
    0x443F64EC5A371195ULL, 0x4112CF68649A260EULL, 0xD813F2FAB7F5C5CAULL,
    0x660D3257380841EEULL, 0x59AC2C7873F910A3ULL, 0xE846963877671A17ULL,
    0x93B633ABFA3469F8ULL, 0xC0C0F5A60EF4CDCFULL, 0xCAF21ECD4377B28CULL,
    0x57277707199B8175ULL, 0x506C11B9D90E8B1DULL, 0xD83CC2687A19255FULL,
    0x4A29C6465A314CD1ULL, 0xED2DF21216235097ULL, 0xB5635C95FF7296E2ULL,
    0x22AF003AB672E811ULL, 0x52E762596BF68235ULL, 0x9AEBA33AC6ECC6B0ULL,
    0x944F6DE09134DFB6ULL, 0x6C47BEC883A7DE39ULL, 0x6AD047C430A12104ULL,
    0xA5B1CFDBA0AB4067ULL, 0x7C45D833AFF07862ULL, 0x5092EF950A16DA0BULL,
    0x9338E69C052B8E7BULL, 0x455A4B4CFE30E3F5ULL, 0x6B02E63195AD0CF8ULL,
    0x6B17B224BAD6BF27ULL, 0xD1E0CCD25BB9C169ULL, 0xDE0C89A556B9AE70ULL,
    0x50065E535A213CF6ULL, 0x9C1169FA2777B874ULL, 0x78EDEFD694AF1EEDULL,
    0x6DC93D9526A50E68ULL, 0xEE97F453F06791EDULL, 0x32AB0EDB696703D3ULL,
    0x3A6853C7E70757A7ULL, 0x31865CED6120F37DULL, 0x67FEF95D92607890ULL,
    0x1F2B1D1F15F6DC9CULL, 0xB69E38A8965C6B65ULL, 0xAA9119FF184CCCF4ULL,
    0xF43C732873F24C13ULL, 0xFB4A3D794A9A80D2ULL, 0x3550C2321FD6109CULL,
    0x371F77E76BB8417EULL, 0x6BFA9AAE5EC05779ULL, 0xCD04F3FF001A4778ULL,
    0xE3273522064480CAULL, 0x9F91508BFFCFC14AULL, 0x049A7F41061A9E60ULL,
    0xFCB6BE43A9F2FE9BULL, 0x08DE8A1C7797DA9BULL, 0x8F9887E6078735A1ULL,
    0xB5B4071DBFC73A66ULL, 0x230E343DFBA08D33ULL, 0x43ED7F5A0FAE657DULL,
    0x3A88A0FBBCB05C63ULL, 0x21874B8B4D2DBC4FULL, 0x1BDEA12E35F6A8C9ULL,
    0x53C065C6C8E63528ULL, 0xE34A1D250E7A8D6BULL, 0xD6B04D3B7651DD7EULL,
    0x5E90277E7CB39E2DULL, 0x2C046F22062DC67DULL, 0xB10BB459132D0A26ULL,
    0x3FA9DDFB67E2F199ULL, 0x0E09B88E1914F7AFULL, 0x10E8B35AF3EEAB37ULL,
    0x9EEDECA8E272B933ULL, 0xD4C718BC4AE8AE5FULL, 0x81536D601170FC20ULL,
    0x91B534F885818A06ULL, 0xEC8177F83F900978ULL, 0x190E714FADA5156EULL,
    0xB592BF39B0364963ULL, 0x89C350C893AE7DC1ULL, 0xAC042E70F8B383F2ULL,
    0xB49B52E587A1EE60ULL, 0xFB152FE3FF26DA89ULL, 0x3E666E6F69AE2C15ULL,
    0x3B544EBE544C19F9ULL, 0xE805A1E290CF2456ULL, 0x24B33C9D7ED25117ULL,
    0xE74733427B72F0C1ULL, 0x0A804D18B7097475ULL, 0x57E3306D881EDB4FULL,
    0x4AE7D6A36EB5DBCBULL, 0x2D8D5432157064C8ULL, 0xD1E649DE1E7F268BULL,
    0x8A328A1CEDFE552CULL, 0x07A3AEC79624C7DAULL, 0x84547DDC3E203C94ULL,
    0x990A98FD5071D263ULL, 0x1A4FF12616EEFC89ULL, 0xF6F7FD1431714200ULL,
    0x30C05B1BA332F41CULL, 0x8D2636B81555A786ULL, 0x46C9FEB55D120902ULL,
    0xCCEC0A73B49C9921ULL, 0x4E9D2827355FC492ULL, 0x19EBB029435DCB0FULL,
    0x4659D2B743848A2CULL, 0x963EF2C96B33BE31ULL, 0x74F85198B05A2E7DULL,
    0x5A0F544DD2B1FB18ULL, 0x03727073C2E134B1ULL, 0xC7F6AA2DE59AEA61ULL,
    0x352787BAA0D7C22FULL, 0x9853EAB63B5E0B35ULL, 0xABBDCDD7ED5C0860ULL,
    0xCF05DAF5AC8D77B0ULL, 0x49CAD48CEBF4A71EULL, 0x7A4C10EC2158C4A6ULL,
    0xD9E92AA246BF719EULL, 0x13AE978D09FE5557ULL, 0x730499AF921549FFULL,
    0x4E4B705B92903BA4ULL, 0xFF577222C14F0A3AULL, 0x55B6344CF97AAFAEULL,
    0xB862225B055B6960ULL, 0xCAC09AFBDDD2CDB4ULL, 0xDAF8E9829FE96B5FULL,
    0xB5FDFC5D3132C498ULL, 0x310CB380DB6F7503ULL, 0xE87FBB46217A360EULL,
    0x2102AE466EBB1148ULL, 0xF8549E1A3AA5E00DULL, 0x07A69AFDCC42261AULL,
    0xC4C118BFE78FEAAEULL, 0xF9F4892ED96BD438ULL, 0x1AF3DBE25D8F45DAULL,
    0xF5B4B0B0D2DEEEB4ULL, 0x962ACEEFA82E1C84ULL, 0x046E3ECAAF453CE9ULL,
    0xF05D129681949A4CULL, 0x964781CE734B3C84ULL, 0x9C2ED44081CE5FBDULL,
    0x522E23F3925E319EULL, 0x177E00F9FC32F791ULL, 0x2BC60A63A6F3B3F2ULL,
    0x222BBFAE61725606ULL, 0x486289DDCC3D6780ULL, 0x7DC7785B8EFDFC80ULL,
    0x8AF38731C02BA980ULL, 0x1FAB64EA29A2DDF7ULL, 0xE4D9429322CD065AULL,
    0x9DA058C67844F20CULL, 0x24C0E332B70019B0ULL, 0x233003B5A6CFE6ADULL,
    0xD586BD01C5C217F6ULL, 0x5E5637885F29BC2BULL, 0x7EBA726D8C94094BULL,
    0x0A56A5F0BFE39272ULL, 0xD79476A84EE20D06ULL, 0x9E4C1269BAA4BF37ULL,
    0x17EFEE45B0DEE640ULL, 0x1D95B0A5FCF90BC6ULL, 0x93CBE0B699C2585DULL,
    0x65FA4F227A2B6D79ULL, 0xD5F9E858292504D5ULL, 0xC2B5A03F71471A6FULL,
    0x59300222B4561E00ULL, 0xCE2F8642CA0712DCULL, 0x7CA9723FBB2E8988ULL,
    0x2785338347F2BA08ULL, 0xC61BB3A141E50E8CULL, 0x150F361DAB9DEC26ULL,
    0x9F6A419D382595F4ULL, 0x64A53DC924FE7AC9ULL, 0x142DE49FFF7A7C3DULL,
    0x0C335248857FA9E7ULL, 0x0A9C32D5EAE45305ULL, 0xE6C42178C4BBB92EULL,
    0x71F1CE2490D20B07ULL, 0xF1BCC3D275AFE51AULL, 0xE728E8C83C334074ULL,
    0x96FBF83A12884624ULL, 0x81A1549FD6573DA5ULL, 0x5FA7867CAF35E149ULL,
    0x56986E2EF3ED091BULL, 0x917F1DD5F8886C61ULL, 0xD20D8C88C8FFE65FULL,
    0x31D71DCE64B2C310ULL, 0xF165B587DF898190ULL, 0xA57E6339DD2CF3A0ULL,
    0x1EF6E6DBB1961EC9ULL, 0x70CC73D90BC26E24ULL, 0xE21A6B35DF0C3AD7ULL,
    0x003A93D8B2806962ULL, 0x1C99DED33CB890A1ULL, 0xCF3145DE0ADD4289ULL,
    0xD0E4427A5514FB72ULL, 0x77C621CC9FB3A483ULL, 0x67A34DAC4356550BULL,
    0xF8D626AAAF278509ULL
};
const int piece_offset = 0;
const int castle_offset = 768;
const int ep_offset = 772;
const int turn_offset = 780;
uint64_t book_hash(position_t* pos)
{
    uint64_t hash = 0;
    for (square_t sq=A1; sq<=H8; ++sq) {
        if (!valid_board_index(sq) || pos->board[sq] == EMPTY) continue;
        int index = 64*book_piece_index[pos->board[sq]] +
            8*square_rank(sq) + square_file(sq);
        hash ^= book_random[piece_offset + index];
    }
    if (has_oo_rights(pos, WHITE)) hash ^= book_random[castle_offset + 0];
    if (has_ooo_rights(pos, WHITE)) hash ^= book_random[castle_offset + 1];
    if (has_oo_rights(pos, BLACK)) hash ^= book_random[castle_offset + 2];
    if (has_ooo_rights(pos, BLACK)) hash ^= book_random[castle_offset + 3];
    if (pos->ep_square != EMPTY) {
        hash ^= book_random[ep_offset + square_file(pos->ep_square)];
    }
    if (pos->side_to_move == WHITE) hash ^= book_random[turn_offset];
    return hash;
}
#define my_ntohll(x) \
    (!big_endian ? \
    ((((uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
     (((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
     (((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
     (((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
     (((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
     (((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
     (((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
     (((uint64_t)(x) & 0x00000000000000ffULL) << 56)) : \
     (x))
#define my_ntohs(x) \
    (!big_endian ? \
    ((((uint16_t)(x) & 0xff00) >>  8) | \
     (((uint16_t)(x) & 0x00ff) <<  8)) : \
     (x))
void read_book_entry(int index, book_entry_t* entry)
{
    fseek(book, index * 16, SEEK_SET);
    fread(entry, 16, 1, book);
    entry->key = my_ntohll(entry->key);
    entry->move = my_ntohs(entry->move);
    entry->weight = my_ntohs(entry->weight);
    entry->learn = my_ntohl(entry->learn);
}
int find_book_key(uint64_t target_key)
{
    int high = num_entries, low = -1, mid = 0;
    book_entry_t entry;

    // Since the positions are all in sorted order, just binary search to find
    // the target key.
    while (low < high) {
        mid = (high + low) / 2;
        read_book_entry(mid, &entry);
        if (target_key <= entry.key) high = mid;
        else low = mid + 1;
    }
    read_book_entry(low, &entry);
    assert(high == low);
    return entry.key == target_key ? low : -1;
}
move_t book_move_to_move(position_t* pos, uint16_t book_move)
{
    square_t to = create_square(book_move & 0x7, (book_move>>3) & 0x07);
    square_t from = create_square((book_move>>6) & 0x7, (book_move>>9) & 0x07);
    piece_type_t promote_type = book_move>>12;
    if (promote_type) promote_type++;

    move_t possible_moves[256];
    int num_moves = generate_legal_moves(pos, possible_moves);
    for (int i=0; i<num_moves; ++i) {
        move_t move = possible_moves[i];
        if (is_move_castle(move)) {
            if (is_move_castle_long(move) &&
                    from == get_move_from(move) &&
                    to == (square_t)(queen_rook_home + A8*pos->side_to_move)) {
                return move;
            } else if (is_move_castle_short(move) &&
                    from == get_move_from(move) &&
                    to == (square_t)(king_rook_home + A8*pos->side_to_move)) {
                return move;
            }
            continue;
        }
        if ((piece_type_t)get_move_promote(move) == promote_type &&
                from == get_move_from(move) &&
                to == get_move_to(move)) return move;
    }
    return NO_MOVE;
}
move_t get_poly_book_move(position_t* pos)
{
    uint64_t key = book_hash(pos);
    int offset = find_book_key(key);
    if (offset == -1) return NO_MOVE;

    move_t moves[255];
    uint16_t weights[255];
    uint16_t total_weight = 0;
    int index = 0;
    book_entry_t entry;
    // Read all book entries with the correct key. They're all stored
    // contiguously, so just scan through as long as the key matches.
    while (true) {
        assert(offset+index < num_entries);
        read_book_entry(offset+index, &entry);
        if (entry.key != key) break;
        moves[index] = book_move_to_move(pos, entry.move);
        printf("info string book move ");
        print_coord_move(moves[index]);
        printf("weight %d\n", entry.weight);
        weights[index++] = total_weight + entry.weight;
        total_weight += entry.weight;
    }
    if (index == 0) return NO_MOVE;

    // Choose randomly amonst the weighted options.
    uint16_t choice = random_32() % total_weight;
    int i;
    for (i=0; choice >= weights[i]; ++i) {}
    assert(i < index);
    return moves[i];
}
static void handle_book_file(void* opt, char* value)
{
    uci_option_t* option = opt;
    strncpy(option->value, value, 128);
    int name_len = strlen(value);
    if (value[name_len-3] == 'c' &&
            value[name_len-2] == 't' &&
            value[name_len-1] == 'g') {
        options.book_loaded = init_ctg_book(option->value);
        options.probe_book = &get_ctg_book_move;
    } else {
        options.book_loaded = init_poly_book(option->value);
        options.probe_book = &get_poly_book_move;
    }
}
/*
void init_uci_options()
{
    add_uci_option("Hash", OPTION_SPIN, "64",
            1, 4096, NULL, NULL, &handle_hash);
    add_uci_option("Clear Hash", OPTION_BUTTON, "",
            0, 0, NULL, NULL, &handle_clear_hash);
    add_uci_option("Ponder", OPTION_CHECK, "false",
            0, 0, NULL, &options.ponder, &default_handler);
    add_uci_option("MultiPV", OPTION_SPIN, "1",
            1, 256, NULL, &options.multi_pv, &default_handler);
    add_uci_option("OwnBook", OPTION_CHECK, "false",
            0, 0, NULL, &options.use_book, &default_handler);
    add_uci_option("Book file", OPTION_STRING, "book.bin",
            0, 0, NULL, NULL, &handle_book_file);
    add_uci_option("UCI_Chess960", OPTION_CHECK, "false",
            0, 0, NULL, &options.chess960, &default_handler);
    add_uci_option("Arena-style 960 castling", OPTION_CHECK, "false",
            0, 0, NULL, &options.arena_castle, &default_handler);
    add_uci_option("Use Gaviota tablebases", OPTION_CHECK, "false",
            0, 0, NULL, &options.use_gtb, &handle_gtb_use);
    add_uci_option("Gaviota tablebase path", OPTION_STRING, ".",
            0, 0, NULL, NULL, &handle_gtb_path);
    char* schemes[6] = { "uncompressed", "cp1", "cp2", "cp3", "cp4", NULL };
    add_uci_option("Gaviota compression scheme", OPTION_COMBO, "cp4",
            0, 0, schemes, &options.gtb_scheme, &handle_gtb_scheme);
    add_uci_option("Gaviota tablebase cache size", OPTION_SPIN, "32",
            0, 4096, NULL, &options.gtb_cache_size, &handle_gtb_cache);
    add_uci_option("Load tablebases in a separate thread", OPTION_CHECK, "true",
            0, 0, NULL, &options.nonblocking_gtb, &default_handler);
    add_uci_option("Tablebase pieces", OPTION_SPIN, "5",
            3, 6, NULL, &options.max_egtb_pieces, &default_handler);
    add_uci_option("Use Scorpio bitbases", OPTION_CHECK, "false",
            0, 0, NULL, &options.use_scorpio_bb, &handle_scorpio_bb_use);
    add_uci_option("Scorpio bitbase path", OPTION_STRING, ".",
            0, 0, NULL, NULL, &handle_scorpio_bb_path);
    add_uci_option("Pawn cache size", OPTION_SPIN, "1",
            1, 128, NULL, NULL, &handle_pawn_cache);
    add_uci_option("PV cache size", OPTION_SPIN, "32",
            1, 1024, NULL, NULL, &handle_pv_cache);
    add_uci_option("Output Delay", OPTION_SPIN, "2000",
            0, 1000000, NULL, &options.output_delay, &default_handler);
    char* verbosities[4] = { "low", "medium", "high", NULL };
    add_uci_option("Verbosity", OPTION_COMBO, "low",
            0, 0, verbosities, &options.verbosity, &handle_verbosity);
    options.book_loaded = false;
}
*/

void init_daydreamer(void)
{
    // Figure out if we're on a big- or little-endian system.
    const int i = 1;
    big_endian = (*(char*)&i) == 0;

    init_hash();
    init_material_table(4*1024*1024);
    init_bitboards();
    generate_attack_data();
    init_eval();
    //init_uci_options();
    set_position(&root_data.root_pos, FEN_STARTPOS);
}
int main(int argc, char *argv[])
{
    int i, j, count;
    search_data_t root_shits;

    init_daydreamer();

    //print_board(&fuckeverything, false);

    init_search_data(&root_shits);
    for (i = 0; i < atoi(argv[1]); i++) {
        set_position(&root_shits.root_pos,
                "5k1K/8/8/8/8/8/8/q7 w - - 4 36");
        printf("find_checks: %d\n", find_checks(&root_shits.root_pos));
        generate_legal_moves(&root_shits.root_pos, moves);
    }

    for (count = 0; moves[count] != 0; count++);
    printf("legal moves: %d\n", count);
    printf("is_check: %d\n", is_check(&root_shits.root_pos));

    return 0;
}
