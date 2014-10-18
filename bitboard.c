#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

#include "bitboard.h"
#include "magicmoves.h"
#include "common.h"

static uint64_t knight_occupancy_mask[] = {
    0x0000000000020400, 0x0000000000050800, 0x00000000000a1100, 0x0000000000142200, 
    0x0000000000284400, 0x0000000000508800, 0x0000000000a01000, 0x0000000000402000, 
    0x0000000002040004, 0x0000000005080008, 0x000000000a110011, 0x0000000014220022, 
    0x0000000028440044, 0x0000000050880088, 0x00000000a0100010, 0x0000000040200020, 
    0x0000000204000402, 0x0000000508000805, 0x0000000a1100110a, 0x0000001422002214, 
    0x0000002844004428, 0x0000005088008850, 0x000000a0100010a0, 0x0000004020002040, 
    0x0000020400040200, 0x0000050800080500, 0x00000a1100110a00, 0x0000142200221400, 
    0x0000284400442800, 0x0000508800885000, 0x0000a0100010a000, 0x0000402000204000, 
    0x0002040004020000, 0x0005080008050000, 0x000a1100110a0000, 0x0014220022140000, 
    0x0028440044280000, 0x0050880088500000, 0x00a0100010a00000, 0x0040200020400000, 
    0x0204000402000000, 0x0508000805000000, 0x0a1100110a000000, 0x1422002214000000, 
    0x2844004428000000, 0x5088008850000000, 0xa0100010a0000000, 0x4020002040000000, 
    0x0400040200000000, 0x0800080500000000, 0x1100110a00000000, 0x2200221400000000, 
    0x4400442800000000, 0x8800885000000000, 0x100010a000000000, 0x2000204000000000, 
    0x0004020000000000, 0x0008050000000000, 0x00110a0000000000, 0x0022140000000000, 
    0x0044280000000000, 0x0088500000000000, 0x0010a00000000000, 0x0020400000000000, 
};

static uint64_t king_occupancy_mask[] = {
    0x0000000000000302, 0x0000000000000705, 0x0000000000000e0a, 0x0000000000001c14, 
    0x0000000000003828, 0x0000000000007050, 0x000000000000e0a0, 0x000000000000c040, 
    0x0000000000030203, 0x0000000000070507, 0x00000000000e0a0e, 0x00000000001c141c, 
    0x0000000000382838, 0x0000000000705070, 0x0000000000e0a0e0, 0x0000000000c040c0, 
    0x0000000003020300, 0x0000000007050700, 0x000000000e0a0e00, 0x000000001c141c00, 
    0x0000000038283800, 0x0000000070507000, 0x00000000e0a0e000, 0x00000000c040c000, 
    0x0000000302030000, 0x0000000705070000, 0x0000000e0a0e0000, 0x0000001c141c0000, 
    0x0000003828380000, 0x0000007050700000, 0x000000e0a0e00000, 0x000000c040c00000, 
    0x0000030203000000, 0x0000070507000000, 0x00000e0a0e000000, 0x00001c141c000000, 
    0x0000382838000000, 0x0000705070000000, 0x0000e0a0e0000000, 0x0000c040c0000000, 
    0x0003020300000000, 0x0007050700000000, 0x000e0a0e00000000, 0x001c141c00000000, 
    0x0038283800000000, 0x0070507000000000, 0x00e0a0e000000000, 0x00c040c000000000, 
    0x0302030000000000, 0x0705070000000000, 0x0e0a0e0000000000, 0x1c141c0000000000, 
    0x3828380000000000, 0x7050700000000000, 0xe0a0e00000000000, 0xc040c00000000000, 
    0x0203000000000000, 0x0507000000000000, 0x0a0e000000000000, 0x141c000000000000, 
    0x2838000000000000, 0x5070000000000000, 0xa0e0000000000000, 0x40c0000000000000, 
};

static const unsigned char BitsSetTable256[256] = 
{
#define B2(n) n,     n+1,     n+1,     n+2
#define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
#define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
    B6(0), B6(1), B6(1), B6(2)
#undef B2
#undef B4
#undef B6
};

static u64 *fen_to_bitboard(char c, board_t *b)
{
    struct bitboard *ptr;

    ptr = isupper(c) ? &b->white_pieces : &b->black_pieces;

    c = tolower(c);

    switch (c) {
        case 'p':
            return &ptr->pawns;
            break;
        case 'n':
            return &ptr->knights;
            break;
        case 'b':
            return &ptr->bishops;
            break;
        case 'r':
            return &ptr->rooks;
            break;
        case 'q':
            return &ptr->queens;
            break;
        case 'k':
            return &ptr->king;
            break;
        default:
            return NULL; // error
            break;
    }
}

void init_bitboards(char *_fen, board_t *board)
{
    int i, j;
    int col, row;
    char *fen_ptr, *rank, *tmp;
    struct bitboard *w, *b;
    u64 *tmpbb;

    if (_fen == NULL)
        return;

    char fen[strlen(_fen) + 1];
    strcpy(fen, _fen);

    fen_ptr = strchr(fen, ' ');
    *fen_ptr++ = 0;

    // initialize board
    rank = fen;
    tmp = strchr(fen, '/');
    *tmp++ = 0;
    row = 7;
    for (i = 0; i < 8; i++) {
        col = 7;
        for (j = 0; j < strlen(rank); j++) {
            if (isdigit((int)rank[j])) {
                int cnt = 0;
                for (; cnt < rank[j] - '0'; --col, ++cnt);
            } else {
                tmpbb = fen_to_bitboard(rank[j], board);
                set_bit(*tmpbb, coord_to_index(row, col));
                col--;
            }
        }
        row--;

        rank = tmp;
        if (tmp) {
            tmp = strchr(tmp, '/');
            if (tmp) {
                *tmp++ = 0;
            }
        }
    }

    w = &board->white_pieces;
    w->pieces = w->pawns | w->rooks | w->knights
        | w->bishops | w->queens | w->king;

    b = &board->black_pieces;
    b->pieces = b->pawns | b->rooks | b->knights
        | b->bishops | b->queens | b->king;

    w->apieces = b->apieces = w->pieces | b->pieces;

    board->turn = *fen_ptr == 'w' ? WHITE : BLACK;
}

/* https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable */
static inline int bits_set(uint64_t v)
{
    unsigned int c; // c is the total bits set in v

    c = BitsSetTable256[v & 0xff] + 
        BitsSetTable256[(v >> 8) & 0xff] + 
        BitsSetTable256[(v >> 16) & 0xff] + 
        BitsSetTable256[(v >> 24) & 0xff] +
        BitsSetTable256[(v >> 32) & 0xff] +
        BitsSetTable256[(v >> 40) & 0xff] +
        BitsSetTable256[(v >> 48) & 0xff] +
        BitsSetTable256[(v >> 56) & 0xff];

    return c;
}

static inline void get_set_bits(uint64_t n, int *arr)
{
    int i, j;

    j = 0;
    for (i = 0; i < 64; i++) {
        if ((n >> i) & 1)
            arr[j++] = i;
    }
    arr[j] = -1;
}

static void index_to_coords(coord_t *c, int index)
{
    // 0 == H1
    // 7 == A1
    // 8 == H2
    c->y = index / 8;
    c->x = index % 8;
}

static int lsb_to_index(u64 board)
{
    int i;

    if (!board)
        return -1;

    for (i = 0; i < 64; i++)
        if (board & (1lu << i))
            return i;

    return -1;
}

int bb_can_attack(u64 moves, int pos)
{
    int i;

    for (i = 0; i < 64; i++)
        if (moves & (1lu << i) && i == pos)
            return 1;
    return 0;
}

void bb_print(u64 b)
{
    int i, j;

    for (i = 7; i >= 0; i--) {
        for (j = 7; j >= 0; j--)
            printf("%lu ", ((b >> (i * 8 + j) & 1)));
        putchar('\n');
    }
    printf("0x%016lx\n\n", b);
}

static inline u64 generate_rook_moves(struct bitboard *b, int pos)
{
    return Rmagic(pos, b->apieces & magicmoves_r_mask[pos]) & ~b->pieces;
}

static inline u64 generate_bishop_moves(struct bitboard *b, int pos)
{
    return Bmagic(pos, b->apieces & magicmoves_b_mask[pos]) & ~b->pieces;
}

static inline u64 generate_queen_moves(struct bitboard *b, int pos)
{
    return Qmagic(pos, b->apieces & (magicmoves_r_mask[pos] | magicmoves_b_mask[pos])) & ~b->pieces;
}

static inline u64 generate_knight_moves(struct bitboard *b, int pos)
{
    return knight_occupancy_mask[pos] & ~b->pieces;
}

static inline u64 generate_king_moves(struct bitboard *b, int pos)
{
    return king_occupancy_mask[pos] & ~b->pieces;
}

/* returns moves, initializes *attacks */
static inline u64 generate_pawn_moves(struct bitboard *b, int pos,
        u64 *attacks, int turn)
{
    u64 ret;
    static const uint64_t filea = 0x7f7f7f7f7f7f7f7f;
    static const uint64_t fileh = 0xfefefefefefefefe;

    if (turn == WHITE) {
        ret = (b->pawns << 8) & ~b->pieces;
        ret |= ((ret & 0xff0000) << 8) & ~b->pieces;
        *attacks = (b->pawns & fileh) << 7;
        *attacks |= (b->pawns & filea) << 9;
    } else {
        ret = (b->pawns >> 8) & ~b->pieces;
        ret |= ((ret & 0x0000ff0000000000) >> 8) & ~b->pieces;
        *attacks = (b->pawns & filea) >> 7;
        *attacks |= (b->pawns & fileh) >> 9;
    }
    *attacks &= (b->apieces & ~b->pieces);

    return ret;
}

static void store_pawn_moves(struct bitboard *board, u64 moves, u64 attacks,
        struct move *move_arr, int *num_moves, int turn)
{
    int bits[65], i, count = *num_moves;
    static const uint64_t filea = 0x7f7f7f7f7f7f7f7f;
    static const uint64_t fileh = 0xfefefefefefefefe;

    get_set_bits(board->pawns, bits);
    for (i = 0; bits[i] != -1; i++) {
        // turn is -1 for black, 1 for white
        if (moves & (1lu << (bits[i] + 8 * turn))) {
            index_to_coords(&move_arr[count].frm, bits[i]);
            index_to_coords(&move_arr[count].to, bits[i] + 8 * turn);
            count++;

            if (moves & (1lu << (bits[i] + 16 * turn))) {
                index_to_coords(&move_arr[count].frm, bits[i]);
                index_to_coords(&move_arr[count].to, bits[i] + 16 * turn);
                count++;
            }
        }

        if (attacks & ((1lu << (bits[i] + 7 * turn)) & filea)) {
            index_to_coords(&move_arr[count].frm, bits[i]);
            index_to_coords(&move_arr[count].to, bits[i] + 7 * turn);
            count++;
        }

        if (attacks & ((1lu << (bits[i] + 9 * turn)) & fileh)) {
            index_to_coords(&move_arr[count].frm, bits[i]);
            index_to_coords(&move_arr[count].to, bits[i] + 9 * turn);
            count++;
        }
    }
    *num_moves = count;
}

static inline void store_moves(u64 moves, int pos,
        struct move *move_arr, int *num_moves)
{
    int j, count = *num_moves;

    for (j = 0; moves && j < 64; j++)
        if (moves & (1lu << j)) {
            index_to_coords(&move_arr[count].frm, pos);
            index_to_coords(&move_arr[count].to, j);
            ++count;

            moves &= ~(1lu << j);
        }
    *num_moves = count;
}

/* returns the board where bit number 'pos' is set */
static inline u64 *find_board(struct bitboard *b, int pos)
{
    if (is_set(b->pawns, pos))
        return &b->pawns;
    else if (is_set(b->rooks, pos))
        return &b->rooks;
    else if (is_set(b->knights, pos))
        return &b->knights;
    else if (is_set(b->bishops, pos))
        return &b->bishops;
    else if (is_set(b->queens, pos))
        return &b->queens;
    else if (is_set(b->king, pos))
        return &b->king;

    return NULL;
}

int bb_do_move(board_t *board, int index)
{
    struct bitboard *enemy, *self;
    struct move *m;
    int to, from;
    u64 *move_board, *capture_board;

    if (index > board->moves_count || index < 0) {
        debug_print("error: index (%d) > board->moves_count || %d < 0\n",
                index, index);
        return 0;
    }

    m = &board->moves[index];
    to = coord_to_index(m->to.y, m->to.x);
    from = coord_to_index(m->frm.y, m->frm.x);

    if (board->turn == WHITE) {
        enemy = &board->black_pieces;
        self = &board->white_pieces;
    } else {
        enemy = &board->white_pieces;
        self = &board->black_pieces;
    }

    board->backup.move_board = move_board = find_board(self, from);
    if (move_board == NULL) {
        debug_print("failed to obtain move board. board is likely not a board\n");
        return 0;
    }

    board->backup.capture_board = capture_board = find_board(enemy, to);
    if (capture_board != NULL) {
        board->backup.capture_mask = isolate_bit(*capture_board, to);
        clear_bit(*capture_board, to);
    } else
        board->backup.capture_mask = 0;

    clear_bit(*move_board, from);
    set_bit(*move_board, to);

    // update friends and "all pieces"-board
    self->pieces = self->pawns | self->rooks | self->knights
        | self->bishops | self->queens | self->king;
    enemy->pieces = enemy->pawns | enemy->rooks | enemy->knights
        | enemy->bishops | enemy->queens | enemy->king;
    self->apieces = enemy->apieces = self->pieces | enemy->pieces;

    // check for pawn promotion
    if (&self->pawns == move_board) {
        if ((board->turn == WHITE && m->to.y == 7) ||
                (board->turn == BLACK && m->to.y == 0)) {
            board->backup.promotion = 1;
            clear_bit(*move_board, to);
            set_bit(self->queens, to);
        }
    }

    return 1;
}

int bb_undo_move(board_t *board, int index)
{
    struct bitboard *enemy, *self;
    struct move *m;
    int to, from;
    u64 *move_board, *capture_board, mask;

    if (index > board->moves_count)
        return 0;

    m = &board->moves[index];
    to = coord_to_index(m->to.y, m->to.x);
    from = coord_to_index(m->frm.y, m->frm.x);

    move_board = board->backup.move_board;
    capture_board = board->backup.capture_board;
    mask = board->backup.capture_mask;

    if (board->turn == WHITE) {
        enemy = &board->black_pieces;
        self = &board->white_pieces;
    } else {
        enemy = &board->white_pieces;
        self = &board->black_pieces;
    }

    clear_bit(*move_board, to);
    set_bit(*move_board, from);

    if (capture_board)
        set_bit(*capture_board, mask);

    // update friends and "all pieces"-board
    self->pieces = self->pawns | self->rooks | self->knights
        | self->bishops | self->queens | self->king;
    enemy->pieces = enemy->pawns | enemy->rooks | enemy->knights
        | enemy->bishops | enemy->queens | enemy->king;
    self->apieces = enemy->apieces = self->pieces | enemy->pieces;

    // check for pawn promotion
    if (board->backup.promotion)
        clear_bit(self->queens, to);

    return 1;
}

/* checks if current player is in check.
 */
int bb_calculate_check(board_t *board)
{
    u64 attacks;
    struct bitboard *enemy, *self;
    int bits[65], i, king_idx;

    if (board->turn == WHITE) {
        enemy = &board->black_pieces;
        self = &board->white_pieces;
    } else {
        enemy = &board->white_pieces;
        self = &board->black_pieces;
    }
    king_idx = lsb_to_index(self->king);

    get_set_bits(enemy->king, bits);
    for (i = 0; bits[i] != -1; i++)
        if (bb_can_attack(generate_king_moves(enemy, bits[i]), king_idx))
            goto yes;

    get_set_bits(enemy->queens, bits);
    for (i = 0; bits[i] != -1; i++) {
        if (bb_can_attack(generate_queen_moves(enemy, bits[i]), king_idx))
            goto yes;
    }

    get_set_bits(enemy->rooks, bits);
    for (i = 0; bits[i] != -1; i++)
        if (bb_can_attack(generate_rook_moves(enemy, bits[i]), king_idx))
            goto yes;

    get_set_bits(enemy->bishops, bits);
    for (i = 0; bits[i] != -1; i++)
        if (bb_can_attack(generate_bishop_moves(enemy, bits[i]), king_idx))
            goto yes;

    get_set_bits(enemy->knights, bits);
    for (i = 0; bits[i] != -1; i++)
        if (bb_can_attack(generate_knight_moves(enemy, bits[i]), king_idx))
            goto yes;

    generate_pawn_moves(enemy, bits[i], &attacks, -board->turn);
    if (bb_can_attack(attacks, king_idx))
        goto yes;

    board->is_check = 0;
    return 0;
yes:
    board->is_check = 1;
    return 1;
}

void bb_generate_all_legal_moves(board_t *board)
{
    u64 moves, attacks;
    struct bitboard *b;
    int bits[65], i;

    b = board->turn == WHITE ? &board->white_pieces : &board->black_pieces;

    get_set_bits(b->king, bits);
    for (i = 0; bits[i] != -1; i++)
        store_moves(generate_king_moves(b, bits[i]), bits[i],
                board->moves, &board->moves_count);

    get_set_bits(b->queens, bits);
    for (i = 0; bits[i] != -1; i++)
        store_moves(generate_queen_moves(b, bits[i]), bits[i],
                board->moves, &board->moves_count);

    get_set_bits(b->rooks, bits);
    for (i = 0; bits[i] != -1; i++)
        store_moves(generate_rook_moves(b, bits[i]), bits[i],
                board->moves, &board->moves_count);

    get_set_bits(b->bishops, bits);
    for (i = 0; bits[i] != -1; i++)
        store_moves(generate_bishop_moves(b, bits[i]), bits[i],
                board->moves, &board->moves_count);

    get_set_bits(b->knights, bits);
    for (i = 0; bits[i] != -1; i++) {
        store_moves(generate_knight_moves(b, bits[i]), bits[i],
                board->moves, &board->moves_count);
    }

    moves = generate_pawn_moves(b, bits[i], &attacks, board->turn);
    store_pawn_moves(b, moves, attacks, board->moves,
            &board->moves_count, board->turn);
}
