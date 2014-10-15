#include <stdint.h>
#include "common.h"
#include "fastrules.h"
#include "board.h"

extern void init_daydreamer(void);
extern int32_t *please_give_me_all_legal_moves_remember_to_free_this_structure_after(char *fen);

static int initialized = 0;

void init(void)
{
    init_daydreamer();
    initialized = 1;
}

board_t *get_all_legal_moves(board_t *board_struct)
{
    int i;
    char *fen;
    int32_t *moves;
    unsigned char from, to;

    if (!initialized)
        init();

    fen = get_fen(board_struct);
    
    moves = please_give_me_all_legal_moves_remember_to_free_this_structure_after(fen);

    free(fen);
    if(moves[0] == 0xffffffff)
        board_struct->moves_count = -1;
    else{
        for (i = 0; moves[i] != 0; i++) {
            from = moves[i] & 0xff;
            to = (moves[i] >> 8) & 0xff;
    
            board_struct->moves[i].frm.x = from & 0xf;
            board_struct->moves[i].frm.y = from >> 4;
    
            board_struct->moves[i].to.x = to & 0xf;
            board_struct->moves[i].to.y = to >> 4;
        }
    
        board_struct->moves_count = i;
    }
    free(moves);
    return board_struct;
}
