#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include<pthread.h>

#include "common.h"
#include "board.h"
#include "bitboard.h"

static piece_t fen_to_chesspiece(char c)
{
    switch (c) {
        case 'P':
            return WHITE_PAWN;
            break;
        case 'N':
            return WHITE_KNIGHT;
            break;
        case 'B':
            return WHITE_BISHOP;
            break;
        case 'R':
            return WHITE_ROOK;
            break;
        case 'Q':
            return WHITE_QUEEN;
            break;
        case 'K':
            return WHITE_KING;
            break;
        case 'p':
            return BLACK_PAWN;
            break;
        case 'n':
            return BLACK_KNIGHT;
            break;
        case 'b':
            return BLACK_BISHOP;
            break;
        case 'r':
            return BLACK_ROOK;
            break;
        case 'q':
            return BLACK_QUEEN;
            break;
        case 'k':
            return BLACK_KING;
            break;
        default:
            return -1; // error
            break;
    }
}

static char chesspiece_to_fen(piece_t c)
{
    switch (c) {
        case WHITE_PAWN:
            return 'P';
            break;
        case WHITE_KNIGHT:
            return 'N';
            break;
        case WHITE_BISHOP:
            return 'B';
            break;
        case WHITE_ROOK:
            return 'R';
            break;
        case WHITE_QUEEN:
            return 'Q';
            break;
        case WHITE_KING:
            return 'K';
            break;
        case BLACK_PAWN:
            return 'p';
            break;
        case BLACK_KNIGHT:
            return 'n';
            break;
        case BLACK_BISHOP:
            return 'b';
            break;
        case BLACK_ROOK:
            return 'r';
            break;
        case BLACK_QUEEN:
            return 'q';
            break;
        case BLACK_KING:
            return 'k';
            break;
        default:
            return -1; // error
            break;
    }
}

board_t *new_board(char *_fen)
{
    int i, j;
    int col, row;
    board_t *board;
    char *fen_ptr, *rank, *tmp;

    if (_fen == NULL || *_fen == 0)
        _fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    char fen[strlen(_fen) + 1];

    strcpy(fen, _fen);

    board = calloc(1, sizeof(board_t));
    if (board == NULL)
        return NULL;

    for (i = 0; i < 8; i++)
        board->board_2d[i] = &board->_board[i * 8];
    board->board = board->_board; // backwards compatability


    fen_ptr = strchr(fen, ' ');
    *fen_ptr++ = 0;

    // initialize board
    rank = fen;
    tmp = strchr(fen, '/');
    *tmp++ = 0;
    row = 7;
    for (i = 0; i < 8; i++) {
        col = 0;
        for (j = 0; j < strlen(rank); j++) {
            if (isdigit((int)rank[j])) {
                int cnt = 0;
                for (; cnt < rank[j] - '0'; ++col, ++cnt)
                    board->board_2d[row][col] = P_EMPTY;
            } else {
                board->board_2d[row][col] = fen_to_chesspiece(rank[j]);
                col++;
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
    
    board->is_check = -1;
    board->moves_count = -1;
    board->turn = *fen_ptr == 'w' ? WHITE : BLACK;

    init_bitboards(_fen, board);

    return board;
}

void free_board(board_t *b)
{
    free(b);
}

void generate_all_moves(board_t *b)
{
    // already calculated moves for this round
    if (b->moves_count != -1)
        return;

    b->moves_count = 0;
    bb_generate_all_legal_moves(b);
}

int is_check(board_t *board)
{
    if (board->is_check == -1)
        bb_calculate_check(board);

    return board->is_check;
}

int is_stalemate(board_t *b)
{
    generate_all_moves(b);
    return b->moves_count == 0;
}

int is_checkmate(board_t *b)
{
    return is_stalemate(b) && is_check(b);
}

void swapturn(board_t *b)
{
    b->turn = -b->turn;
    b->is_check = -1;
    b->moves_count = -1;
}

static inline void del_move(board_t *b, int n)
{
    if (--b->moves_count != -1)
        b->moves[n] = b->moves[b->moves_count];
}

int undo_move(board_t *b, int n)
{
    struct move *m;
    int tx, fx, ret;

    m = &b->moves[n];
    // convert coordinates from bitboard coords (0->7, 1->6, ..., 7->0)
    tx = ~m->to.x & 0x7;
    fx = ~m->frm.x & 0x7;

#ifdef BOARD_CONSISTENCY_TEST
    printf("%s: consistency check before move has been undone\n", __func__);
    printf("%s: move was from %d, %d to %d, %d\n", __func__, m->frm.y, fx, m->to.y, tx);
    board_consistency_check(b);
#endif

    b->is_check = -1;

    ret = bb_undo_move(b, n);

    PIECE(b->board, m->frm.y, fx) = PIECE(b->board, m->to.y, tx);
    PIECE(b->board, m->to.y, tx) = b->backup.piece;

    if (b->backup.promotion) {
        PIECE(b->board, m->frm.y, fx) = b->turn == WHITE
            ? WHITE_PAWN : BLACK_PAWN;
    }

    switch (ret) {
        case 3: // move was white short castling
            PIECE(b->board, 0, 7) = PIECE(b->board, 0, 5);
            PIECE(b->board, 0, 5) = P_EMPTY;
            break;
        case 4: // move was white long castling
            PIECE(b->board, 0, 0) = PIECE(b->board, 0, 3);
            PIECE(b->board, 0, 3) = P_EMPTY;
            break;
        case 5: // move was black short castling
            PIECE(b->board, 7, 7) = PIECE(b->board, 7, 5);
            PIECE(b->board, 7, 5) = P_EMPTY;
            break;
        case 6: // move was black long castling
            PIECE(b->board, 7, 0) = PIECE(b->board, 7, 3);
            PIECE(b->board, 7, 3) = P_EMPTY;
            break;
        case 7: // move was en passant
            PIECE(b->board, m->to.y, tx) = P_EMPTY;
            PIECE(b->board, m->to.y - 1 * b->turn, tx) = b->backup.piece;
            break;
        default:
            break;
    }

#ifdef BOARD_CONSISTENCY_TEST
    printf("%s: consistency check after move has been undone\n", __func__);
    printf("%s: move was from %d, %d to %d, %d\n", __func__, m->frm.y, fx, m->to.y, tx);
    board_consistency_check(b);
#endif
    return 1;
}
//pthread_mutex_t lock;

/* should only be called from UCI mode. It does not verify that the move did not
 * put the player in check */
int do_actual_move(board_t *b, struct move *m, struct uci *iface)
{
//                    pthread_mutex_lock(&lock);

    int tx, fx, ret;

    // convert coordinates from bitboard coords (0->7, 1->6, ..., 7->0)
    tx = bb_col_to_AI_col(m->to.x);
    fx = bb_col_to_AI_col(m->frm.x);

#ifdef BOARD_CONSISTENCY_TEST
    printf("%s: consistency check before move has been done\n", __func__);
    printf("%s: move is from %d, %d to %d, %d\n", __func__, m->frm.y, fx, m->to.y, tx);
    print_board(b->board);
    bb_print(b->white_pieces.apieces);
    board_consistency_check(b);
#endif

    b->is_check = -1;
    ret = bb_do_actual_move(b, m);
    if (ret == 0) {
        struct bitboard *bb = b->turn == WHITE ? &b->white_pieces :
            &b->black_pieces;
        printf("FATAL FUCKING ERROR: %s: SOMETHING WENT WRONG\n", __FUNCTION__);
        printf("%s: HALTING EXECUTION!!1\n", __FUNCTION__);
        printf("position: %s\n", iface->position);
        print_board(b->board);
        bb_print(bb->apieces);
        asm("int3");
        return 0;
    } else if (ret == 2) // attempted castling move was illegal
        return 0;

    b->backup.piece = PIECE(b->board, m->to.y, tx);
    PIECE(b->board, m->to.y, tx) = PIECE(b->board, m->frm.y, fx);
    PIECE(b->board, m->frm.y, fx) = P_EMPTY;

    switch (ret) {
        case 3: // move was white short castling
            PIECE(b->board, 0, 5) = PIECE(b->board, 0, 7);
            PIECE(b->board, 0, 7) = P_EMPTY;
            break;
        case 4: // move was white long castling
            PIECE(b->board, 0, 3) = PIECE(b->board, 0, 0);
            PIECE(b->board, 0, 0) = P_EMPTY;
            break;
        case 5: // move was black short castling
            PIECE(b->board, 7, 5) = PIECE(b->board, 7, 7);
            PIECE(b->board, 7, 7) = P_EMPTY;
            break;
        case 6: // move was black long castling
            PIECE(b->board, 7, 3) = PIECE(b->board, 7, 0);
            PIECE(b->board, 7, 0) = P_EMPTY;
            break;
        case 7: // en passant
            b->backup.piece = PIECE(b->board, m->to.y - 1 * b->turn, tx);
            PIECE(b->board, m->to.y - 1 * b->turn, tx) = P_EMPTY;
            break;
        default:
            break;
    }

    if (b->backup.promotion) {
        /*
           printf("promotion! turn: %s. promotion character: %c (0x%02x)\n",
           b->turn == WHITE ? "white" : "black", m->promotion, m->promotion);
           getchar();
           */
        switch (tolower(m->promotion)) {
            case 'q':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_QUEEN : BLACK_QUEEN;
                break;
            case 'n':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_KNIGHT : BLACK_KNIGHT;
                break;
            case 'b':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_BISHOP: BLACK_BISHOP;
                break;
            case 'r':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_ROOK: BLACK_ROOK;
                break;
            default:
                // use queen as default promotion if no other option was set
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_QUEEN : BLACK_QUEEN;
                break;
        }
    }

#ifdef BOARD_CONSISTENCY_TEST
    printf("%s: consistency check after move has been done\n", __func__);
    printf("%s: move was from %d, %d to %d, %d\n", __func__, m->frm.y, fx, m->to.y, tx);
    board_consistency_check(b);
#endif
                   // pthread_mutex_unlock(&lock);

    return 1;
}

/* call this if you are totally sure the move is a legal one */
int do_move(board_t *b, int n)
{
    struct move *m;
    int tx, fx, ret;

    if (n >= b->moves_count)
        return 0;

    m = &b->moves[n];
    // convert coordinates from bitboard coords (0->7, 1->6, ..., 7->0)
    tx = bb_col_to_AI_col(m->to.x);
    fx = bb_col_to_AI_col(m->frm.x);

#ifdef BOARD_CONSISTENCY_TEST
    printf("%s: consistency check before move has been done\n", __func__);
    printf("%s: move is from %d, %d to %d, %d\n", __func__, m->frm.y, fx, m->to.y, tx);
    board_consistency_check(b);
#endif

    b->is_check = -1;
    ret = bb_do_move(b, n);
    if (ret == 0) {
        printf("FATAL FUCKING ERROR: %s: SOMETHING WENT WRONG\n", __FUNCTION__);
        printf("%s: HALTING EXECUTION!!1\n", __FUNCTION__);
        printf("b = %p, n = %d, moves_count = %d\n",
                b, n, b->moves_count);
        asm("int3");
        return 0;
    } else if (ret == 2) {
        del_move(b, n);
        return 0;
    }

    if (is_check(b)) {
        bb_undo_move(b, n);
        del_move(b, n);
        b->is_check = -1;
        return 0;
    }

    b->backup.piece = PIECE(b->board, m->to.y, tx);
    PIECE(b->board, m->to.y, tx) = PIECE(b->board, m->frm.y, fx);
    PIECE(b->board, m->frm.y, fx) = P_EMPTY;

    switch (ret) {
        case 3: // move was white short castling
            PIECE(b->board, 0, 5) = PIECE(b->board, 0, 7);
            PIECE(b->board, 0, 7) = P_EMPTY;
            break;
        case 4: // move was white long castling
            PIECE(b->board, 0, 3) = PIECE(b->board, 0, 0);
            PIECE(b->board, 0, 0) = P_EMPTY;
            break;
        case 5: // move was black short castling
            PIECE(b->board, 7, 5) = PIECE(b->board, 7, 7);
            PIECE(b->board, 7, 7) = P_EMPTY;
            break;
        case 6: // move was black long castling
            PIECE(b->board, 7, 3) = PIECE(b->board, 7, 0);
            PIECE(b->board, 7, 0) = P_EMPTY;
            break;
        case 7: // en passant
            b->backup.piece = PIECE(b->board, m->to.y - 1 * b->turn, tx);
            PIECE(b->board, m->to.y - 1 * b->turn, tx) = P_EMPTY;
            break;
        default:
            break;
    }

    if (b->backup.promotion) {
        /*
           printf("promotion! turn: %s. promotion character: %c (0x%02x)\n",
           b->turn == WHITE ? "white" : "black", m->promotion, m->promotion);
           getchar();
           */
        switch (tolower(m->promotion)) {
            case 'q':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_QUEEN : BLACK_QUEEN;
                break;
            case 'n':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_KNIGHT : BLACK_KNIGHT;
                break;
            case 'b':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_BISHOP: BLACK_BISHOP;
                break;
            case 'r':
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_ROOK: BLACK_ROOK;
                break;
            default:
                // use queen as default promotion if no other option was set
                PIECE(b->board, m->to.y, tx) = b->turn == WHITE
                    ? WHITE_QUEEN : BLACK_QUEEN;
                break;
        }
    }

#ifdef BOARD_CONSISTENCY_TEST
    printf("%s: consistency check after move has been done\n", __func__);
    printf("%s: move was from %d, %d to %d, %d\n", __func__, m->frm.y, fx, m->to.y, tx);
    board_consistency_check(b);
#endif
    return 1;
}

/* call this if you are _NOT_ sure the move is a legal one.
 * This is a wrapper function for do_move.
 * returns -1 if there are no more legal moves to do
 * returns 0 if n > b->moves_count
 * returns 1 if a move was successfully taken */
int move(board_t *b, int n)
{
    do {
        if (is_stalemate(b))
            return -1;
        if (n >= b->moves_count)
            return 0;
    } while (do_move(b, n) != 1);

    return 1;
}

static char fen_buffer[1024];
char *get_fen(board_t *board)
{
    int row, col, cnt;
    char *ret = fen_buffer, *ptr;

    ptr = ret;

    cnt = 0;
    for (row = 7; row >= 0; row--) {
        for (col = 0; col < 8; col++) {
            if (board->board_2d[row][col] == P_EMPTY) {
                ++cnt;
            } else {
                if (cnt) {
                    *ptr++ = cnt + '0';
                    cnt = 0;
                }
                *ptr++ = chesspiece_to_fen(board->board_2d[row][col]);
            }
        }
        if (cnt) {
            *ptr++ = cnt + '0';
            cnt = 0;
        }
        *ptr++ = '/';
    }

    *(ptr - 1) = ' ';
    *ptr++ = board->turn == WHITE ? 'w' : 'b';
    *ptr++ = ' ';
    *ptr = 0;

    if (!board->white_pieces.king_has_moved) {
        if (!board->white_pieces.short_rook_moved)
            *ptr++ = 'K';
        if (!board->white_pieces.long_rook_moved)
            *ptr++ = 'Q';
        *ptr = ' ';
    }

    if (!board->black_pieces.king_has_moved) {
        if (!board->black_pieces.short_rook_moved)
            *ptr++ = 'k';
        if (!board->black_pieces.long_rook_moved)
            *ptr++ = 'q';
        *ptr = ' ';
    }

    if (*ptr == 0)
        strcat(ptr, "- -");
    else {
        ptr[1] = '-';
        ptr[2] = 0;
    }

    return ret;
}

const char *piece_to_str(piece_t p)
{
    const char *ret = NULL;
    static const char *strings[] = {
        "pawn(w)",
        "rook(w)",
        "knight(w)",
        "bishop(w)",
        "queen(w)",
        "king(w)",

        "pawn(b)",
        "rook(b)",
        "knight(b)",
        "bishop(b)",
        "queen(b)",
        "king(b)",
        "empty",
    };

    if (p & (1 << 12))
        ret = strings[12];
    else if (p > 1 << 5) {
        ret = strings[get_moves_index(p) + 6];
    } else  if (p > 0)
        ret = strings[get_moves_index(p) + 0];
    return ret;
}

void print_board(piece_t *board)
{
    int i, j;

    printf("           0         1         2         3"
            "         4         5         6         7\n");

    for (i = 7; i >= 0; i--) {
        printf("%d  ", i + 1);
        for (j = 0; j < 8; j++)
            printf("%10s", piece_to_str(PIECE(board, i, j)));
        printf("    %d\n", i);
    }
    printf("           a         b         c         d"
            "         e         f         g         h\n");
    printf("\n");
}

void print_move(board_t *board, int n)
{
    printf("%02d: (%d, %d) -> (%d, %d)\n", n, board->moves[n].frm.y, ~board->moves[n].frm.x & 0x7,
            board->moves[n].to.y, ~board->moves[n].to.x & 0x7);
}

void print_legal_moves(board_t *board)
{
    int i;

    printf("count: %d\n", board->moves_count);

    for (i = 0; i < board->moves_count; i++)
        printf("%02d: (%d, %d) -> (%d, %d)\n", i, board->moves[i].frm.y, ~board->moves[i].frm.x & 0x7,
                board->moves[i].to.y, ~board->moves[i].to.x & 0x7);
}

static void internal_notation_to_uci(char *u, move_t *m)
{
    static int lookup[] = {'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'};

    u[0] = lookup[m->frm.x];
    u[1] = m->frm.y + '1';
    u[2] = lookup[m->to.x];
    u[3] = m->to.y + '1';
    u[4] = m->promotion;
    u[5] = 0;
}

static void uci_move_notation_to_internal(char *u, move_t *m)
{
    u[0] = tolower(u[0]) - 'a';
    u[1] = u[1] - '1';
    u[2] = tolower(u[2]) - 'a';
    u[3] = u[3] - '1';

    m->frm.x = AI_col_to_bb_col(u[0]);
    m->frm.y = u[1];
    m->to.x = AI_col_to_bb_col((int)u[2]);
    m->to.y = u[3];

    m->promotion = u[4];
}

int do_uci_move(board_t *board, struct uci *iface)
{
    struct move m;
    int fx, tx;
    char *move, move_copy[32];
    
    move = uci_get_next_move(iface);
    strncpy(move_copy, move, sizeof(move_copy));
    //print_board(board->board);
     //   printf("got: %s\n",move);

    memset(&m, 0, sizeof(m));
    uci_move_notation_to_internal(move_copy, &m);
    uci_register_new_move(iface, move);

    fx = bb_col_to_AI_col(m.frm.x);
    tx = bb_col_to_AI_col(m.to.x);
    if ((PIECE(board->board, m.frm.y, fx) == WHITE_PAWN ||
                PIECE(board->board, m.frm.y, fx) == BLACK_PAWN) &&
            PIECE(board->board, m.to.y, tx) == P_EMPTY) {
        m.en_passant = 1;
    }

    do_actual_move(board, &m, iface);

    if (is_stalemate(board))
        return 0;
    if (is_checkmate(board))
        return -1;

    swapturn(board);
    return 1;
}

void register_move_to_uci(struct move *m, struct uci *iface)
{
    char uci[32];

    internal_notation_to_uci(uci, m);
    uci_register_new_move(iface, uci);
    uci_start_search(iface);
}

#ifdef BOARD_CONSISTENCY_TEST
static void consistency_error(char *color, char *piece, int y, int x,
        board_t *board, u64 bitboard)
{
    printf("board consistency check failed! %s %s on "
            "AI board is not set on bitboard (%d, %d)\n",
            color, piece, y, x);
    print_board(board->board);
    bb_print(bitboard);
    printf("Press a key to continue check..\n");
    getchar();
}

void board_consistency_check(board_t *board)
{
    int i, j, num_errors;
    struct bitboard *bb, *bb2;
    char *tmp;
    u64 tmpboard;

    num_errors = 0;

    bb = &board->white_pieces;
    bb2 = &board->white_pieces;
    tmpboard = bb->pawns & bb->rooks & bb->knights & bb->bishops & bb->queens & bb->king &
        bb2->pawns & bb2->rooks & bb2->knights & bb2->bishops & bb2->queens & bb2->king;
    if (tmpboard) {
        printf("error: multiple pieces in same square!:\n");
        bb_print(tmpboard);
        printf("black pawns:\n");
        bb_print(bb->pawns);
        printf("black rooks:\n");
        bb_print(bb->rooks);
        printf("black knights:\n");
        bb_print(bb->pawns);
        printf("black bishops:\n");
        bb_print(bb->bishops);
        printf("black queens:\n");
        bb_print(bb->queens);
        printf("black king:\n");
        bb_print(bb->king);

        printf("white pawns:\n");
        bb_print(bb2->pawns);
        printf("white rooks:\n");
        bb_print(bb2->rooks);
        printf("white knights:\n");
        bb_print(bb2->pawns);
        printf("white bishops:\n");
        bb_print(bb2->bishops);
        printf("white queens:\n");
        bb_print(bb2->queens);
        printf("white king:\n");
        bb_print(bb2->king);
        printf("Press a key to continue check..\n");
        getchar();
        ++num_errors;
    }

    for (i = 7; i >= 0; i--) {
        for (j = 0; j < 8; j++) {
            if (PIECE(board->board, i, j) > WHITE_KING) {
                bb = &board->black_pieces;
                tmp = "black";
            } else {
                bb = &board->white_pieces;
                tmp = "white";
            }

            switch (PIECE(board->board, i, j)) {
                case WHITE_PAWN:
                case BLACK_PAWN:
                    if (!is_set(bb->pawns, AI_coord_to_index(i, j))) {
                        consistency_error(tmp, "pawn", i, j, board, bb->pawns);
                        ++num_errors;
                    }
                    break;
                case WHITE_ROOK:
                case BLACK_ROOK:
                    if (!is_set(bb->rooks, AI_coord_to_index(i, j))) {
                        consistency_error(tmp, "rook", i, j, board, bb->rooks);
                        ++num_errors;
                    }
                    break;
                case WHITE_KNIGHT:
                case BLACK_KNIGHT:
                    if (!is_set(bb->knights, AI_coord_to_index(i, j))) {
                        consistency_error(tmp, "knight", i, j, board, bb->knights);
                        ++num_errors;
                    }
                    break;
                case WHITE_BISHOP:
                case BLACK_BISHOP:
                    if (!is_set(bb->bishops, AI_coord_to_index(i, j))) {
                        consistency_error(tmp, "bishop", i, j, board, bb->bishops);
                        ++num_errors;
                    }
                    break;
                case WHITE_QUEEN:
                case BLACK_QUEEN:
                    if (!is_set(bb->queens, AI_coord_to_index(i, j))) {
                        consistency_error(tmp, "queen", i, j, board, bb->queens);
                        ++num_errors;
                    }
                    break;
                case WHITE_KING:
                case BLACK_KING:
                    if (!is_set(bb->king, AI_coord_to_index(i, j))) {
                        consistency_error(tmp, "king", i, j, board, bb->king);
                        ++num_errors;
                    }
                    break;
                case P_EMPTY:
                    if (is_set(bb->apieces, AI_coord_to_index(i, j))) {
                        consistency_error("empty", "square", i, j, board, board->white_pieces.apieces);
                        ++num_errors;
                    }
                    break;
            }
        }
    }
    if (num_errors == 0)
        printf("No errors\n");
    else
        printf("Total %d errors\n", num_errors);
}
#endif
