#ifndef __UCI_H
#define __UCI_H

struct uci {
    int out_fds[2], in_fds[2];
    FILE *in, *out;
    char *position;
    char *pos_end; // points to the end of the position string
    int position_size;
    int pid;

    unsigned depth; // depth the engine will search
    unsigned search_time; // engime search time in ms
    char __next_move[1024];
};

#define UCI_DEFAULT_FEN "startpos"

extern struct uci *uci_init(char *path, char *fen, int white);
extern void uci_new_game(struct uci *iface, char *fen);
extern void uci_close(struct uci *iface);
extern char *uci_get_next_move(struct uci *iface);
extern void uci_register_new_move(struct uci *iface, char *move);
extern void uci_start_search(struct uci *iface);

#endif
