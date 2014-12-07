#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "common.h"
#include "board.h"
#include "bitboard.h"

static piece_t fen_to_chesspiece(char c)
{
    switch (c) {
        case 'P':
            return WHITE_PAWN;
        case 'N':
            return WHITE_KNIGHT;
        case 'B':
            return WHITE_BISHOP;
        case 'R':
            return WHITE_ROOK;
        case 'Q':
            return WHITE_QUEEN;
        case 'K':
            return WHITE_KING;
        case 'p':
            return BLACK_PAWN;
        case 'n':
            return BLACK_KNIGHT;
        case 'b':
            return BLACK_BISHOP;
        case 'r':
            return BLACK_ROOK;
        case 'q':
            return BLACK_QUEEN;
        case 'k':
            return BLACK_KING;
        default:
            return (piece_t)-1; // error
    }
}

static char chesspiece_to_fen(piece_t c)
{
    switch (c) {
        case WHITE_PAWN:
            return 'P';
        case WHITE_KNIGHT:
            return 'N';
        case WHITE_BISHOP:
            return 'B';
        case WHITE_ROOK:
            return 'R';
        case WHITE_QUEEN:
            return 'Q';
        case WHITE_KING:
            return 'K';
        case BLACK_PAWN:
            return 'p';
        case BLACK_KNIGHT:
            return 'n';
        case BLACK_BISHOP:
            return 'b';
        case BLACK_ROOK:
            return 'r';
        case BLACK_QUEEN:
            return 'q';
        case BLACK_KING:
            return 'k';
        default:
            return -1; // error
    }
}

static void init_fen(board_t *board, char *fen)
{
    int col, row, i, j;
    char *fen_ptr, *rank, *tmp;

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

    board->turn = *fen_ptr == 'w' ? WHITE : BLACK;
}

void set_board(board_t *board, const char *_fen)
{
    int i;

    if (_fen == NULL || *_fen == 0)
        _fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    char fen[strlen(_fen) + 1];

    piece_t *tmp = board->cu_board;
    bzero(board, sizeof(board_t));
    board->cu_board = tmp;

    for (i = 0; i < 8; i++)
        board->board_2d[i] = &board->_board[i * 8];
    board->board = board->_board; // backwards compatability

    strcpy(fen, _fen);
    init_fen(board, fen);
    
    board->is_check = -1;
    board->moves_count = -1;
    board->turn = WHITE;

    strcpy(fen, _fen);
    init_bitboards(fen, board);
}

board_t *new_board(const char *_fen)
{
    board_t *board;

    board = (board_t *)calloc(1, sizeof(board_t));
    if (board == NULL)
        return NULL;

    cudaMalloc(&board->cu_board, sizeof(piece_t) * 128);

    if (_fen != NULL)
        set_board(board, _fen);

    return board;
}

void free_board(board_t *b)
{
    cudaFree(b->cu_board);
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
    int tx, fx;

    b->is_check = -1;

    bb_undo_move(b, n);

    m = &b->moves[n];
    tx = ~m->to.x & 0x7;
    fx = ~m->frm.x & 0x7;
    PIECE(b->board, m->frm.y, fx) = PIECE(b->board, m->to.y, tx);
    PIECE(b->board, m->to.y, tx) = b->backup.piece;

    if (b->backup.promotion) {
        PIECE(b->board, m->frm.y, fx) = b->turn == WHITE
            ? WHITE_PAWN : BLACK_PAWN;
    }

    return 1;
}

/* call this if you are totally sure the move is a legal one */
int do_move(board_t *b, int n)
{
    struct move *m;
    int tx, fx;

    if (n >= b->moves_count)
        return 0;


    b->is_check = -1;
    if (bb_do_move(b, n) != 1) {
        printf("FATAL FUCKING ERROR: %s: SOMETHING WENT WRONG\n", __FUNCTION__);
        printf("%s: HALTING EXECUTION!!1\n", __FUNCTION__);
        printf("b = %p, n = %d, moves_count = %d\n",
                b, n, b->moves_count);
        asm("int3");
        return 0;
    }

    if (is_check(b)) {
#ifdef DEBUG
        debug_print("%d/%d is an illegal move.. undoing\n    ", n, b->moves_count);
        print_move(b, n);
#endif
        bb_undo_move(b, n);
        del_move(b, n);
        b->is_check = -1;
        return 0;
    }

    m = &b->moves[n];

    debug_print("moving from %d, %d to %d, %d\n", m->frm.y, ~m->frm.x & 0x7,
            m->to.y, ~m->to.x & 0x7);

    tx = ~m->to.x & 0x7;
    fx = ~m->frm.x & 0x7;
    b->backup.piece = PIECE(b->board, m->to.y, tx);
    PIECE(b->board, m->to.y, tx) = PIECE(b->board, m->frm.y, fx);
    PIECE(b->board, m->frm.y, fx) = P_EMPTY;

    if (b->backup.promotion) {
        PIECE(b->board, m->to.y, tx) = b->turn == WHITE
            ? WHITE_QUEEN : BLACK_QUEEN;
    }

    return 1;
}

/* call this if you are _NOT_ sure the move is a legal one.
 * This is a wrapper function for do_move.
 * returns -1 if there is no more legal moves to do
 * returns 0 if n > b->moves_count
 * returns 1 if a move was successfully taken */
int move(board_t *b, int n)
{
    do {
        if (is_stalemate(b))
            return -1;
        if (n >= b->moves_count)
            return 0;
    } while (!do_move(b, n));

    return 1;
}

char *get_fen(board_t *board)
{
    int row, col, cnt;
    char *ret = (char *)malloc(256), *ptr;

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
    *ptr = 0;
    strcat(ptr, " - -");

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
        printf("%d  ", i);
        for (j = 0; j < 8; j++)
            printf("%10s", piece_to_str(PIECE(board, i, j)));
        printf("\n");
    }
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

    for (i = 0; i < board->moves_count; i++) {
        printf("%02d: (%d, %d) -> (%d, %d)\n", i, board->moves[i].frm.y, ~board->moves[i].frm.x & 0x7,
                board->moves[i].to.y, ~board->moves[i].to.x & 0x7);

        /*
           if (empty(board->board, board->moves[i].to.y, board->moves[i].to.x, board->turn))
           printf("(empty)\n");
           else if (enemy(board->board, board->moves[i].to.y, board->moves[i].to.x, board->turn))
           printf("(enemy)\n");
           else
           printf("(ally)\n");
           */
    }

}
