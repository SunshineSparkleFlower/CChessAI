#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include "common.h"
#include "uci.h"

/* if fen is NULL, standard position is assumed.
 * white is 1 if engine is white */
struct uci *uci_init(char *path, char *fen, int color)
{
    struct uci *ret;
    char buffer[2048];

    ret = malloc(sizeof (struct uci));
    if (ret == NULL) {
        perror("malloc");
        return NULL;
    }

    if (pipe(ret->in_fds) == -1 || pipe(ret->out_fds) == -1) {
        perror("pipe");
        return NULL;
    }
    ret->out = fdopen(ret->out_fds[0], "r");
    ret->in = fdopen(ret->in_fds[1], "w");

    setvbuf(ret->out, NULL, _IOLBF, 0);
    setvbuf(ret->in, NULL, _IOLBF, 0);

    if (!(ret->pid = fork())) {
        dup2(ret->out_fds[1], 1);
        dup2(ret->in_fds[0], 0);
        dup2(open("/dev/null", O_RDWR), 2);
        execl(path, path, NULL);
        perror("execl");
        _exit(1);
    }

    fprintf(ret->in, "uci\n");
    while (fgets(buffer, sizeof buffer, ret->out)) {
#ifdef UCI_DEBUG
        printf("%s", buffer);
#endif
        if (!strcmp(buffer, "uciok\n"))
            break;
    }
    // fprintf(ret->in, "setoption name Skill Level value %d\n", 0);
    //  fprintf(ret->in, "setoption name Contempt Factor value %d\n", 100);
    //fprintf(ret->in, "setoption name Ponder value %d\n", 0);
    //fprintf(ret->in, "setoption name UCI_Chess960 value True\n");

    fprintf(ret->in, "isready\n");
    fprintf(ret->in, "ucinewgame\n");

    if (fgets(buffer, sizeof buffer, ret->out) == NULL) {
        printf("failed to read from engine\n");
        perror("fgets");
        return NULL;
    }
    if (strcmp(buffer, "readyok\n")) {
        printf("%s responded '%s' to isready!\n", path, buffer);
        free(ret);
        return NULL;
    }
#ifdef UCI_DEBUG
    printf("%s", buffer);
#endif

    ret->position_size = 4096;
    ret->position = malloc(ret->position_size);
    if (ret->position == NULL) {
        free(ret);
        return NULL;
    }

    uci_new_game(ret, fen);

    ret->depth = 1;
    ret->search_time = 10;



    return ret;
}

void uci_new_game(struct uci *iface, char *fen)
{

    if (fen == NULL)
        fen = "startpos";

    //sprintf(iface->position, "position %s moves \n", fen);
    iface->pos_end = iface->position + strlen(iface->position) - 1;

}

void uci_close(struct uci *iface)
{
    fprintf(iface->in, "stop\n");
    fprintf(iface->in, "quit\n");
    close(iface->in_fds[0]);
    close(iface->in_fds[1]);
    close(iface->out_fds[0]);
    close(iface->out_fds[1]);
    free(iface->position);
    waitpid(iface->pid, NULL, 0);
    free(iface);
}
// returns the next move in a statically allocated string buffer

char *uci_get_next_move(struct uci *iface)
{

    char *tmp;
    //fprintf(iface->in, "stop\n");
    //printf("stop\n");

    while (fgets(iface->__next_move, sizeof (iface->__next_move), iface->out)) {
#ifdef UCI_DEBUG
        printf("got: %s", iface->__next_move);
#endif
        if ((tmp = strstr(iface->__next_move, "bestmove ")))
            break;
    }
    if (tmp == NULL)
        return NULL; // why would this ever happen?

    tmp = strchr(iface->__next_move, ' ') + 1;
    *(strchr(tmp, ' ')) = 0;

    return tmp;
}

void uci_register_new_move(struct uci *iface, char *move)
{

    if ((iface->pos_end - iface->position + strlen(move) + 1) >
            iface->position_size) {
        iface->position_size *= 2;
        iface->position = realloc(iface->position, iface->position_size);
        if (iface->position == NULL) {
            perror("realloc");
            exit(1);
        }
        iface->pos_end = iface->position + strlen(iface->position);
    }
    strcpy(iface->pos_end, move);
    iface->pos_end += strlen(move);

    if (*(iface->pos_end) != '\n') {
        strcpy(iface->pos_end, " \n");
        ++iface->pos_end;
    }

#ifdef UCI_DEBUG
    printf("%s", iface->position);
#endif
}

void uci_start_search(struct uci *iface, board_t *board)
{

    int tmp;
    char buffer[256];
    char fen[1024];

    get_fen(board, fen);
    fprintf(iface->in, "position fen %s - 0 1\n", fen);
    sprintf(buffer, "go depth %d\n", 2);

#ifdef UCI_DEBUG
    printf("%s", buffer);
#endif

    tmp = fprintf(iface->in, "%s", buffer);
    //tmp = fwrite(buffer, 1, strlen(buffer), iface->in);
    if (tmp != strlen(buffer)) {
        printf("failed to write buffer to engine\n");
        perror("fwrite");
    }
    fflush(iface->in);

}
