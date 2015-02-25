#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "threadpool.h"
#include "AI.h"
#include "common.h"
#include "board.h"

struct command {
    char *name;
    int (*callback)(char *params[], int num_params);
};

struct search {
    AI_instance_t *ai;
    int finished;
    struct move m;
};

static struct {
    char *name, *code;
    int debug;

    int num_threads, num_jobs;
// setoption command options
    int ponder;
} settings;

static volatile board_t *board;
static volatile struct search *searches; // each AI thread should have its own structure
static struct job *jobs;

static int uci_uci(char *params[], int num_params)
{
    printf("id name %s\n", ENGINE_NAME);
    printf("id author %s\n", ENGINE_AUTHOR);
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

static int uci_setoption(char *params[], int num_params)
{
    if (!strcmp(params[1], "Ponder")) {
        settings.ponder = !strcmp(params[3], "false") ? 0 : 1;
    }
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

/* convert from UCI move notation to internal move struct */
static void notation_to_move(char *move_notation, struct move *m)
{
    coord_t *to, *from;

    from = &m->frm;
    to = &m->to;

    from->x = tolower(move_notation[0]) - 'a';
    from->y = move_notation[1] - '1';

    to->x = tolower(move_notation[2]) - 'a';
    to->y = move_notation[3] - '1';
}

/* convert from internal move struct to UCI notation */
static void move_to_notation(char *move_notation, struct move *m)
{
    coord_t *to, *from;

    from = &m->frm;
    to = &m->to;

    move_notation[0] = from->x + 'a';
    move_notation[1] = from->y + '1';

    move_notation[2] = to->x + 'a';
    move_notation[3] = to->y + '1';

    move_notation[4] = 0;
}

static int uci_position(char *params[], int num_params)
{
    int i;
    char *fen;
    struct move m;

    // initialize fen string
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

    // initialize board
    if (board)
        free_board((board_t *)board);
    board = new_board(fen);

    // find the index of the moves parameters
    for (i = 2; i < num_params; i++)
        if (!strcmp(params[i], "moves"))
            break;

    if (strcmp(params[i], "moves"))
        return 1;

    // to the moves
    for (; params[i]; i++) {
        notation_to_move(params[i], &m);
        do_actual_move((board_t *)board, &m);
    }

    return 1;
}

void AI_search(void *arg)
{
    int index;
    char buffer[16];
    struct search *s = (struct search *)arg;

    fprintf(stderr, "starting search ai @ %p, board @ %p\n", s->ai, board);

    s->finished = 0;
    index = _get_best_move(s->ai, (board_t *)board);
    memcpy((void *)&s->m, (void *)&board->moves[index], sizeof(struct move));
    s->finished = 1;

    fprintf(stderr, "from: %d, %d, to: %d, %d\n",
            board->moves[index].frm.y, board->moves[index].frm.x,
            board->moves[index].to.y, board->moves[index].to.x);

    move_to_notation(buffer, &s->m);
    printf("bestmove %s\n", buffer);
    fprintf(stderr, "bestmove %s\n", buffer);
}

static int uci_go(char *params[], int num_params)
{
    int i;

    generate_all_moves((board_t *)board);
    for (i = 0; i < settings.num_jobs; i++) {
        jobs[i].task = AI_search;
        jobs[i].data = (struct search *)&searches[i];
        put_new_job(&jobs[i]);
    }

    return 0;
}

static int uci_stop(char *params[], int num_params)
{
    char tmp[32];
    /*
    while (!searches[0].finished);

    move_to_notation(tmp, (struct move *)&searches[0].m);
    printf("bestmove %s\n", tmp);
    */

    free_board(board);
    board = new_board(DEFAULT_FEN);

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

static int uci_ponderhit(char *params[], int num_params)
{
    // TODO
    return 0;
}

static int uci_quit(char *params[], int num_params)
{
    exit(0);
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

    fprintf(stderr, "got command. parameters:\n");
    for (i = 0; params[i]; i++)
        fprintf(stderr, "   %d. %s\n", i, params[i]);

    if (!params[0])
        return 1;

    fprintf(stderr, "executing %s\n", params[0]);
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
    int i;

    dup2(open("/tmp/alice.out", O_RDWR), 2);

    fprintf(stderr, "Alice is running\n");

    if (argc != 2) {
        fprintf(stderr, "USAGE: %s <ai-file>\n", argv[0]);
        fprintf(stderr, "argc = %d. %s, %s\n", argc, argv[0], argv[1]);
        exit(0);
    }

    // TODO parse command line arguments

    memset(&settings, 0, sizeof(settings));
    settings.num_jobs = settings.num_threads = 1;

    init_threadpool(settings.num_threads);
    jobs = malloc(settings.num_jobs * sizeof(struct job));

    searches = malloc(settings.num_jobs * sizeof(struct search));

    fprintf(stderr, "initializing ai with file %s. pwd = %s\n", argv[1], get_current_dir_name());
    for (i = 0; i < settings.num_jobs; i++) {
        searches[i].ai = load_ai(argv[1], 1000);
        if (searches[i].ai == NULL) {
            exit(0);
        }
    }
    fprintf(stderr, "initializing board\n");
    board = new_board(DEFAULT_FEN);
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
