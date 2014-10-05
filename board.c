#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "common.h"

#include "fastrules.h"
#include "rules.h"

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

    board = malloc(sizeof(board_t));
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
    
    board->moves_count = 0;
    board->turn = *fen_ptr == 'w' ? WHITE : BLACK;

    return board;
}

void free_board(board_t *b)
{
    free(b);
}

int have_lost(board_t *b)
{
    get_all_legal_moves(b);
    return b->moves_count == 0;
}

void swapturn(board_t *b)
{
    b->turn = -b->turn;
}

void move(board_t *b, int n)
{
    piece_t backup;
    do_move(b->board, b->moves[n].frm, b->moves[n].to, &backup);
}

char *get_fen(board_t *board)
{
    int row, col, cnt;
    char *ret = malloc(256), *ptr;

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
}

void print_legal_moves(board_t *board)
{
    int i;

    printf("count: %d\n", board->moves_count);

    for (i = 0; i < board->moves_count; i++) {
        printf("(%d, %d) -> (%d, %d)\n", board->moves[i].frm.y, board->moves[i].frm.x,
                board->moves[i].to.y, board->moves[i].to.x);

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
