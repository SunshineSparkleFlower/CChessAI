#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>

#include "common.h"
#include "rules.h"

static void add_to_moves(board_t *board, coord_t *from, coord_t *to)
{
    int count = board->moves_count;
    board->moves[count].frm.x = from->x;
    board->moves[count].frm.y = from->y;

    board->moves[count].to.x = to->x;
    board->moves[count].to.y = to->y;

    board->moves_count++;
}

static void remove_from_moves(board_t *board, int index)
{
    board->moves[index].frm =
        board->moves[board->moves_count - 1].frm;

    board->moves[index].to =
        board->moves[board->moves_count - 1].to;

    board->moves_count--;

}

static int legal_pos(coord_t *p)
{
    return p->x < 8 && p->x >= 0 && p->y < 8 && p->y >= 0;
}

void do_move(piece_t *board, coord_t frm, coord_t to, piece_t *backup)
{
    *backup = PIECE(board, to.y, to.x);
    PIECE(board, to.y, to.x) = PIECE(board, frm.y, frm.x);
    PIECE(board, frm.y, frm.x) = P_EMPTY;
}

static coord_t find_king(board_t *board_struct)
{  
    int row, col;
    coord_t coord = {-1, -1};
    piece_t *board = board_struct->board;
    for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
            if (color(PIECE(board, row, col)) == board_struct->turn && get_piece_type(PIECE(board, row, col)) == KING ) {
                coord.y = row;
                coord.x = col;
                return coord;
            }
        }
    }

    return coord; // should never happen
}

static int move_exist(coord_t coord, board_t *board_struct)
{
    int i;
    for (i = 0; i < board_struct->moves_count; i++)
        if (coord.x == board_struct->moves[i].to.x && coord.y == board_struct->moves[i].to.y)
            return 1;

    return 0;
}

static int _get_legal_moves(board_t *board_struct, coord_t *from)
{
    piece_t *board = board_struct->board;

    coord_t tmp, ntmp;
    piece_t piece = PIECE(board, from->y, from->x);
    //   printf("\in from->y:%d, from->x:%d \n", from->y, from->x);
    //   printf("\piece = %d \n", piece);

    int move_offset_index = get_moves_index(piece), i, j;

    //printf("in %s. from: %d, %d\n", __FUNCTION__, from->y, from->x);

    if (move_offset_index > 5 && move_offset_index != 12) {
        printf("ERROR: moves index > 5: %d\n", move_offset_index);

        return -1;
    }


    if (get_piece_type(piece) == EMPTY)
        return 0;

    for (i = 0; !(move_offset[move_offset_index][i][0].x == 0 &&
                move_offset[move_offset_index][i][0].y == 0); i++) {
        for (j = 0; !(move_offset[move_offset_index][i][j].x == 0 &&
                    move_offset[move_offset_index][i][j].y == 0); j++) {
            if (get_piece_type(piece) == PAWN) {
                tmp.x = move_offset[move_offset_index][i][j].x * color(piece);
                tmp.y = move_offset[move_offset_index][i][j].y * color(piece);

                ntmp.x = from->x + tmp.x;
                ntmp.y = from->y + tmp.y;


                if (!legal_pos(&ntmp))
                    continue;

                if (!(move_offset[move_offset_index][i][j].x == 0 ||
                            move_offset[move_offset_index][i][j].y == 0)) {
                    if (enemy(board, ntmp.y, ntmp.x, board_struct->turn))
                        add_to_moves(board_struct, from, &ntmp);
                } else if (empty(board, ntmp.y, ntmp.x, board_struct->turn)) {
                    add_to_moves(board_struct, from, &ntmp);

                }
                continue;
            }

            tmp.x = from->x + move_offset[move_offset_index][i][j].x;
            tmp.y = from->y + move_offset[move_offset_index][i][j].y;

            if (!legal_pos(&tmp))
                continue;

            if (empty(board, tmp.y, tmp.x, board_struct->turn)) {
                add_to_moves(board_struct, from, &tmp);
                continue;
            } else if (enemy(board, tmp.y, tmp.x, board_struct->turn))
                add_to_moves(board_struct, from, &tmp);

            break;
        }
    }

    if (get_piece_type(piece) == PAWN) {
        if (color(piece) == BLACK && from->y == 6) {
            tmp.x = from->x;
            tmp.y = from->y - 2;

            if (empty(board, from->y - 1, from->x, board_struct->turn) &&
                    empty(board, from->y - 2, from->x, board_struct->turn))
                add_to_moves(board_struct, from, &tmp);
        } else if (color(piece) == WHITE && from->y == 1) {
            tmp.x = from->x;
            tmp.y = from->y + 2;

            if (empty(board, from->y + 1, from->x, board_struct->turn) &&
                    empty(board, from->y + 2, from->x, board_struct->turn))
                add_to_moves(board_struct, from, &tmp);
        }
    }

    //printf("Returning from %s\n", __FUNCTION__);
    return board_struct->moves_count;
}

static int get_possible_moves(board_t *board, coord_t *from)
{
    int ret;

    ret = _get_legal_moves(board, from);

    return ret;
}

static board_t *get_all_possible_moves(board_t *board_struct)
{
    int row, col;
    coord_t coord;
    piece_t *board = board_struct->board;

    board_struct->moves_count = 0;
    for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
            if (color(PIECE(board, row, col)) == board_struct->turn) {
                coord.y = row;
                coord.x = col;
                get_possible_moves(board_struct, &coord);
            }
        }
    }

    return board_struct;
}

// checks if 'turn' is in check
int is_check(int turn, board_t *board_struct)
{
    board_t tmp_bs = {.board = board_struct->board, .turn = -turn, .moves_count = 0};
    get_all_possible_moves(&tmp_bs);

    coord_t king_coord = find_king(board_struct);
    return move_exist(king_coord, &tmp_bs);
}

static void rm_checkmoves(board_t *board_struct)
{
    int i, in_chess;
    piece_t backup;
    piece_t *board = board_struct->board;

    for (i = 0; i < board_struct->moves_count; i++) {
        do_move(board, board_struct->moves[i].frm, board_struct->moves[i].to, &backup);

        in_chess = is_check(board_struct->turn, board_struct);

        // move the pieces back to their original position
        PIECE(board, board_struct->moves[i].frm.y, board_struct->moves[i].frm.x) = 
            PIECE(board, board_struct->moves[i].to.y, board_struct->moves[i].to.x);
        PIECE(board, board_struct->moves[i].to.y, board_struct->moves[i].to.x) = backup;

        if (in_chess) {
            debug_print("removing a move\n");

            remove_from_moves(board_struct, i);
            i--;
        }

    }

}

int get_legal_moves(board_t *board, coord_t *from)
{
    int ret;

    ret = _get_legal_moves(board, from);
    debug_print("number of legeal moves in get_legal_moves %d\n", board->moves_count);

    rm_checkmoves(board);
    debug_print("number of legeal moves iafter rm_checkmoves %d\n", board->moves_count);

    return ret;
}

board_t *get_all_legal_moves(board_t *board_struct)
{
    int row, col;
    coord_t coord;
    piece_t *board = board_struct->board;

    board_struct->moves_count = 0;
    for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
            if (color(PIECE(board, row, col)) == board_struct->turn) {
                coord.y = row;
                coord.x = col;
                get_legal_moves(board_struct, &coord);
            }
        }
    }

    return board_struct;
}

static int can_attack(move_t *moves, int num_moves, coord_t *e)
{
    int i;
    for (i = 0; i < num_moves; i++)
        if (moves[i].to.x == e->x && moves[i].to.y == e->y)
            return 1;

    return 0;
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

    for (i = 0; i < 8; i++) {
        printf("%d  ", i);
        for (j = 0; j < 8; j++)
            printf("%10s", piece_to_str(PIECE(board, i, j)));
        printf("\n");
    }
}

static void print_legal_moves(board_t *board)
{
    int i;

    printf("count: %d\n", board->moves_count);
    if (board->moves_count > 0) {
        printf("from: %d, %d (%s)\n", board->moves[0].frm.y, board->moves[0].frm.x,
                piece_to_str(board_2d[board->moves[0].frm.y][board->moves[0].frm.x]));
        piece_to_str(PIECE(board->board, board->moves[0].frm.y, board->moves[0].frm.x));
    }

    for (i = 0; i < board->moves_count; i++) {
        printf("(%d, %d) ", board->moves[i].to.y, board->moves[i].to.x);

        if (empty(board->board, board->moves[i].to.y, board->moves[i].to.x, board->turn))
            printf("(empty)\n");
        else if (enemy(board->board, board->moves[i].to.y, board->moves[i].to.x, board->turn))
            printf("(enemy)\n");
        else
            printf("(ally)\n");
    }

}

int main(int argc, char *argv[])
{
    int i;
    coord_t c;
    board_t testboard;

    if (argc != 3) {
        printf("USAGE: %s <y> <x>\n", argv[0]);
        exit(1);
    }

    //printf("%d, %d: %s\n", c.y, c.x, piece_to_str(board_2d[c.y][c.x]));

    //memcpy(testboard.board, board, sizeof(board));
    testboard.board = board;
    testboard.moves_count = 0;
    testboard.turn = WHITE;

    print_board(testboard.board);

    c.y = atoi(argv[1]);
    c.x = atoi(argv[2]);
    get_legal_moves(&testboard, &c);

    printf("\n");
    printf("%d, %d = %d\n", c.y, c.x, color(board_2d[c.y][c.x]));
    printf("count: %d\n", testboard.moves_count);

    if (testboard.moves_count > 0) {
        printf("from: %d, %d (%s)\n", testboard.moves[0].frm.y, testboard.moves[0].frm.x,
                piece_to_str(board_2d[testboard.moves[0].frm.y][testboard.moves[0].frm.x]));
        piece_to_str(PIECE(testboard.board, testboard.moves[0].frm.y, testboard.moves[0].frm.x));
    }

    for (i = 0; i < testboard.moves_count; i++) {
        printf("(%d, %d) ", testboard.moves[i].to.y, testboard.moves[i].to.x);

        if (empty(testboard.board, testboard.moves[i].to.y, testboard.moves[i].to.x, testboard.turn))
            printf("(empty)\n");
        else if (enemy(testboard.board, testboard.moves[i].to.y, testboard.moves[i].to.x, testboard.turn))
            printf("(enemy)\n");
        else
            printf("(ally)\n");
    }

    printf("is_check: %d\n", is_check(BLACK, &testboard));

    /*
       printf("legal moves from (%d, %d)\n", 3, 1);
       for (i = 0; i < count; i++) {
       printf("(%d, %d) -> (%d, %d)\n", possible_moves[i].from.y, possible_moves[i].from.x,
       possible_moves[i].to.y, possible_moves[i].to.x);
       }
       */



    return 0;
}
