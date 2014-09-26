#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>

#include "common.h"
#include "rules.h"

void print_board(piece_t *board);

/*
static move_t possible_moves[20*16];
static int count = 0; // this must be reset to 0 on each turn-change
static legal_moves_t ret = {possible_moves, 0}; // returned from get_all_legal_moves()
*/

static void add_to_moves(board_t *board, coord_t *from, coord_t *to)
{
    int count = board->moves_count;
    board->moves[count].frm.x = from->x;
    board->moves[count].frm.y = from->y;

    board->moves[count].to.x = to->x;
    board->moves[count].to.y = to->y;

    board->moves_count++;
}

static int legal_pos(coord_t *p)
{
    return p->x < 8 && p->x >= 0 && p->y < 8 && p->y >= 0;
}

static void do_move(piece_t *board, coord_t from, coord_t to, piece_t *backup)
{
    *backup = PIECE(board, to.y, to.x);
    PIECE(board, to.y, to.x) = PIECE(board, from.y, from.x);
}

static void rm_checkmoves(board_t *board_struct)
{
    int i;
    piece_t backup;
    piece_t *board = board_struct->board;

    for (i = 0; i < board_struct->moves_count; i++) {
        do_move(board, board_struct->moves[i].frm, board_struct->moves[i].to, &backup);
        is_check(board_struct->turn, board_struct);

        PIECE(board, board_struct->moves[i].frm.y, board_struct->moves[i].frm.x) = 
            PIECE(board, board_struct->moves[i].to.y, board_struct->moves[i].to.x);
        PIECE(board, board_struct->moves[i].to.y, board_struct->moves[i].to.x) = backup;

        if (!is_check(board_struct->turn, board_struct)) {
            continue;
        }

        board_struct->moves[i].frm =
            board_struct->moves[board_struct->moves_count - 1].frm;
        board_struct->moves[i].to =
            board_struct->moves[board_struct->moves_count - 1].to;
        board_struct->moves_count--;
        i--;
    }

    char buffer[8192] = {0};
    int fd;

    fd = open("/proc/self/maps", O_RDONLY);
    read(fd, buffer, sizeof(buffer));
    printf("%s\n", buffer);

    memset(buffer, 0, sizeof(buffer));
    read(fd, buffer, sizeof(buffer));
    printf("%s\n", buffer);

    memset(buffer, 0, sizeof(buffer));
    read(fd, buffer, sizeof(buffer));
    printf("%s\n", buffer);

    memset(buffer, 0, sizeof(buffer));
    read(fd, buffer, sizeof(buffer));
    printf("%s\n", buffer);

    memset(buffer, 0, sizeof(buffer));
    read(fd, buffer, sizeof(buffer));
    printf("%s\n", buffer);

    memset(buffer, 0, sizeof(buffer));
    read(fd, buffer, sizeof(buffer));
    printf("%s\n", buffer);
    close(fd);

    printf("\nboard_struct @ %p\n", board_struct);
    printf("count = %d\n", board_struct->moves_count);
    board_struct->moves[0].frm.y = 15;
    board_struct->moves_count = 1;
}

int _get_legal_moves(board_t *board_struct, coord_t *from)
{
    coord_t tmp, ntmp;
    piece_t piece = PIECE(board, from->y, from->x);
    int move_offset_index = get_moves_index(piece), i, j;
    piece_t *board = board_struct->board;

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

int get_legal_moves(board_t *board, coord_t *from)
{
    int ret;

    ret = _get_legal_moves(board, from);
    rm_checkmoves(board);

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

// checks if 'player_color' is in check
int is_check(int player_color, board_t *board_struct)
{
    coord_t allies[16];
    int row, col, c, allies_count = 0;
    coord_t enemy_king = {-1, -1}, curr;

    board_t testboard;
    testboard.turn = board_struct->turn;
    testboard.board = board_struct->board;
    //memcpy(testboard.board, board_struct->board, sizeof(testboard.board));
    testboard.moves_count = 0;

    piece_t *board = board_struct->board;

    for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
            c = color(PIECE(board, row, col));
            // is this the enemy king?
            if (c == player_color && get_piece_type(PIECE(board, row, col)) == KING) {
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
                    //printf("%s\n", __FUNCTION__);
                    _get_legal_moves(&testboard, &curr);
                    testboard.moves_count = 0;
                    if (can_attack(testboard.moves, testboard.moves_count, &enemy_king))
                        return 1;
                }
            }
        }
    }

    // check if the remaining allies can attack the enemy king
    for (c = 0; c < allies_count; c++) {
        _get_legal_moves(&testboard, &allies[c]);
        if (can_attack(testboard.moves, testboard.moves_count, &enemy_king))
            return 1;
    }

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

void print_legal_moves(board_t *board)
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
