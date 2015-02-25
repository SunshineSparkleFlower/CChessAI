#include <stdio.h>
#include <string.h>

#include "threadpool.h"
#include "AI.h"
#include "common.h"
#include "board.h"

struct command {
    char *name;
    int (*callback)(char *params[], int num_params);
};

static struct {
    char *name, *code;
    int debug;

    // options that can be set with the "setoption" command
    int num_threads;
} settings;

static board_t *board;

static int uci_uci(char *params[], int num_params)
{
    printf("name %s\n", ENGINE_NAME);
    printf("author %s\n", ENGINE_AUTHOR);
    printf("uciok\n");

    return 0;
}

static int uci_debug(char *params[], int num_params)
{
    if (!strcmp(params[0], "on"))
        settings.debug = 1;
    else if (!strcmp(params[0], "off"))
        settings.debug = 0;

    return 0;
}

/* this is used to synchronize the engine with the GUI. When the GUI has sent a command or
 * multiple commands that can take some time to complete,
 * this command can be used to wait for the engine to be ready again or
 * to ping the engine to find out if it is still alive.
 * E.g. this should be sent after setting the path to the tablebases as this can take some time.
 * This command is also required once before the engine is asked to do any search
 * to wait for the engine to finish initializing.
 * This command must always be answered with "readyok" and can be sent also when the engine is calculating
 * in which case the engine should also immediately answer with "readyok" without stopping the search. */
static int uci_isready(char *params[], int num_params)
{
    printf("readyok\n");

    return 0;
}

static int uci_setoption(char *params[], int num_params)
{
    // TODO
    return 0;
}

static int uci_register(char *params[], int num_params)
{
    int i = 0;
    char buffer[512] = "";

    if (!strcmp(params[0], "later"))
        return 1;

    if (!strcmp(params[0], "name")) {
        if (settings.name)
            free(settings.name);
        for (i = 1; params[i] && !strcmp(params[i], "code"); i++) {
            strncat(buffer, params[i], sizeof(buffer) - strlen(buffer) - 1);
            strncat(buffer, " ", sizeof(buffer) - strlen(buffer) - 1);
        }
        settings.name = strdup(buffer);
    }
    
    if (!strcmp(params[i], "code")) {
        if (settings.name)
            free(settings.name);
        for (i = 1; params[i] && !strcmp(params[i], "name"); i++) {
            strncat(buffer, params[i], sizeof(buffer) - strlen(buffer) - 1);
            strncat(buffer, " ", sizeof(buffer) - strlen(buffer) - 1);
        }
        settings.code = strdup(buffer);
    }

    return 1;
}

/* this is sent to the engine when the next search (started with "position" and "go") will be from
 * a different game. This can be a new game the engine should play or a new game it should analyse but
 * also the next position from a testsuite with positions only.
 * If the GUI hasn't sent a "ucinewgame" before the first "position" command, the engine shouldn't
 * expect any further ucinewgame commands as the GUI is probably not supporting the ucinewgame command.
 * So the engine should not rely on this command even though all new GUIs should support it.
 * As the engine's reaction to "ucinewgame" can take some time the GUI should always send "isready"
 * after "ucinewgame" to wait for the engine to finish its operation. */
static int uci_newgame(char *params[], int num_params)
{
    // WHAT?
    return 0;
}

static int uci_position(char *params[], int num_params)
{
    int i;
    char *fen;

    if (!strcmp(params[0], "startpos"))
        fen = DEFAULT_FEN;
    else {
        char buffer[512] = "";

        for (i = 1; params[i]; i++) {
            if (!strcmp(params[i], "moves"))
                break;
            strncat(buffer, params[i], sizeof(buffer) - strlen(buffer) - 1);
            strncat(buffer, " ", sizeof(buffer) - strlen(buffer) - 1);
        }

        fen = alloca(strlen(buffer) + 1);
        strcpy(fen, buffer);
    }

    if (board)
        free_board(board);

    board = new_board(fen);

    for (i = 2; i < num_params; i++)
        if (!strcmp(params[i], "moves"))
            break;
    if (strcmp(params[i], "moves"))
        return 1;

    // TODO execute moves in list starting at params[i + 1]

    return 1;
}

static int uci_go(char *params[], int num_params)
{

    return 0;
}

static int uci_stop(char *params[], int num_params)
{

    return 0;
}

static int uci_ponderhit(char *params[], int num_params)
{

    return 0;
}

static int uci_quit(char *params[], int num_params)
{

    return 0;
}

int process_command(char *params[], int num_params)
{
    int i;
    static struct command commands[] = {
        {"uci", uci_uci},
        {"debug", uci_debug},
        {"isready", uci_isready},
        {"setoption", uci_setoption},
        {"register", uci_register},
        {"ucinewgame", uci_newgame},
        {"position", uci_position},
        {"go", uci_go},
        {"stop", uci_stop},
        {"ponderhit", uci_ponderhit},
        {"quit", uci_quit},
        {NULL, NULL},
    };

    for (i = 0; commands[i].name; i++)
        if (!strcmp(commands[i].name, params[0]))
            return commands[i].callback(params + 1, num_params - 1);
    return 0;
}

static int parse_input(char *line, char *params[])
{
    int i;
    char *str1, *token, *saveptr1;

    if (line[strlen(line) - 1] == '\n')
        line[strlen(line) - 1] = 0;

    for (i = 0, str1 = line; i < 15; i++, str1 = NULL) {
        token = strtok_r(str1, " ", &saveptr1);
        if (token == NULL)
            break;
        params[i] = token;
    }
    params[i] = NULL;

    return i;
}

static void init(int argc, char **argv)
{
    // TODO parse command line arguments

    memset(&settings, 0, sizeof(settings));
    settings.num_threads = 1;

    init_threadpool(settings.num_threads);
}

int main(int argc, char *argv[])
{
    int num_params;
    char buffer[512], *params[16];

    init(argc, argv);

    while (fgets(buffer, sizeof buffer, stdin)) {
        num_params = parse_input(buffer, params);
        process_command(params, num_params);
    }
    return 0;
}
