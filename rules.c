#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "common.h"

static move_t possible_moves[20*16];
static int count = 0; // this must be reset to 0 on each turn-change
static legal_moves_t ret = {possible_moves, 0}; // returned from get_all_legal_moves()

static void add_to_moves(coord_t *from, coord_t *to)
{
    possible_moves[count].from.x = from->x;
    possible_moves[count].from.y = from->y;

    possible_moves[count].to.x = to->x;
    possible_moves[count].to.y = to->y;

    count++;
}

static int legal_pos(coord_t *p)
{
    return p->x < 8 && p->x >= 0 && p->y < 8 && p->y >= 0;
}

int get_legal_moves(piece_t *board, coord_t *from)
{
    coord_t tmp, ntmp;
    piece_t piece = BOARD(board, from->y, from->x);
    int move_offset_index = get_moves_index(piece), i, j;

    printf("in %s. from: %d, %d\n", __FUNCTION__, from->y, from->x);

    if (move_offset_index > 5) {
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
                    if (enemy(board, ntmp.y, ntmp.x))
                        add_to_moves(from, &ntmp);
                } else if (empty(board, ntmp.y, ntmp.x)) {
                    add_to_moves(from, &ntmp);
                }
                continue;
            }

            tmp.x = from->x + move_offset[move_offset_index][i][j].x;
            tmp.y = from->y + move_offset[move_offset_index][i][j].y;

            if (!legal_pos(&tmp))
                continue;

            if (empty(board, tmp.y, tmp.x)) {
                add_to_moves(from, &tmp);
                continue;
            } else if (enemy(board, tmp.y, tmp.x))
                add_to_moves(from, &tmp);

            break;
        }
    }

    if (get_piece_type(piece) == PAWN) {
        if (color(piece) == BLACK && from->y == 6) {
            tmp.x = from->x;
            tmp.y = from->y - 2;

            if (empty(board, from->y - 1, from->x) &&
                    empty(board, from->y - 2, from->x))
                add_to_moves(from, &tmp);
        } else if (color(piece) == WHITE && from->y == 1) {
            tmp.x = from->x;
            tmp.y = from->y + 2;

            if (empty(board, from->y + 1, from->x) &&
                    empty(board, from->y + 2, from->x))
                add_to_moves(from, &tmp);
        }
    }

    printf("Returning from %s\n", __FUNCTION__);
    return count;
}

legal_moves_t *get_all_legal_moves(piece_t *board)
{
    int row, col;
    coord_t coord;

    count = 0;
    for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
            if (color(BOARD(board, row, col)) == turn) {
                coord.y = row;
                coord.x = col;
                get_legal_moves(board, &coord);
            }
        }
    }

    ret.num_moves = count;

    return &ret;
}

static int can_attack(move_t *moves, int num_moves, coord_t *e)
{
    int i;
    for (i = 0; i < num_moves; i++)
        if (moves[i].to.x == e->x && moves[i].to.y == e->y)
            return 1;

    return 0;
}

// checks if 'player_color' is in check
int is_check(int player_color, piece_t *board)
{
    piece_t *ptr;
    coord_t allies[16];
    int row, col, c, allies_count = 0;
    coord_t enemy_king = {-1, -1}, curr;

    for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
            c = color(BOARD(board, row, col));
            // is this the enemy king?
            if (c == player_color && get_piece_type(BOARD(board, row, col)) == KING) {
                enemy_king.y = row;
                enemy_king.x = col;
            } else if (c == -player_color) { // is this an allie?
                // put the allie into the list of allies to process after the
                // enemy king has been found
                if (enemy_king.y == -1 && enemy_king.x == -1) {
                    allies[allies_count].y = row;
                    allies[allies_count++].x = col;
                } else {
                    // enemy king has been found; check if it can be attacked
                    curr.y = row;
                    curr.x = col;
                    get_legal_moves(board, &curr);
                    count = 0;
                    if (can_attack(possible_moves, count, &enemy_king))
                        return 1;
                }
            }
        }
    }

    // check if the remaining allies can attack the enemy king
    for (c = 0; c < allies_count; c++) {
        get_legal_moves(board, &allies[c]);
        if (can_attack(possible_moves, count, &enemy_king))
            return 1;
    }

    return 0;
}

const char *piece_to_str(piece_t p)
{
    const char *ret;
    static const char *strings[] = {
        "empty",
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
    };

    if (p > 1 << 5) {
        ret = strings[get_moves_index(p) + 7];
    } else  if (p > 0){
        ret = strings[get_moves_index(p) + 1];
    } else
        ret = strings[0];

    return ret;
}

void print_board(piece_t *board)
{
    int i, j;
    piece_t piece;

    printf("           0         1         2         3"
            "         4         5         6         7\n");

    for (i = 0; i < 8; i++) {
        printf("%d  ", i);
        for (j = 0; j < 8; j++)
            printf("%10s", piece_to_str(BOARD(board, i, j)));
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int i;
    coord_t c;

    if (argc != 3) {
        printf("USAGE: %s <y> <x>\n", argv[0]);
        exit(1);
    }

    c.y = atoi(argv[1]);
    c.x = atoi(argv[2]);

    //printf("%d, %d: %s\n", c.y, c.x, piece_to_str(board_2d[c.y][c.x]));

    print_board(board);
    get_legal_moves(board, &c);

    printf("%d, %d = %d\n", c.y, c.x, color(board_2d[c.y][c.x]));
    printf("count: %d\n", count);

    if (count > 0) {
        printf("from: %d, %d (%s)\n", possible_moves[0].from.y, possible_moves[0].from.x,
                piece_to_str(board_2d[possible_moves[0].from.y][possible_moves[0].from.x]));
    }

    for (i = 0; i < count; i++) {
        printf("(%d, %d) ", possible_moves[i].to.y, possible_moves[i].to.x);

        if (empty(board, possible_moves[i].to.y, possible_moves[i].to.x))
            printf("(empty)\n");
        else if (enemy(board, possible_moves[i].to.y, possible_moves[i].to.x))
            printf("(enemy)\n");
        else
            printf("(ally)\n");
    }

    printf("is_check: %d\n", is_check(BLACK, board));

    return 0;
}
