#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <arpa/inet.h>
#include <netdb.h>

#include "board.h"
#include "AI.h"

#define AI_NAME ENGINE_NAME

static int white = 1;
static char ai_file[256] = "./ai_save.aidump";
static char gui_ip[256] = "0";
static int verbose = 0;

static int setup_socket(int port)
{
    int sd, tmp;
    struct sockaddr_in serv_addr;

    sd = socket(AF_INET, SOCK_STREAM, 0);
    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(port);

    tmp = 1;
    setsockopt(sd, SOL_SOCKET, SO_REUSEADDR,(const char *)&tmp, sizeof(int));
    bind(sd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

    listen(sd, 1);

    return sd;
}

static int wait_for_node(int sd)                              
{                                                      
    struct sockaddr_in addr;                           
    socklen_t size = sizeof(struct sockaddr_in);       

    return accept(sd, (struct sockaddr *)&addr, &size);
}                                                      

int node_connect(char *host, int port)
{
    int sd;
    struct hostent *h;
    struct sockaddr_in addr;

    sd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sd == -1)
        return -1;

    h = gethostbyname(host);

    memset(&addr, 0, sizeof(addr));
    memcpy(&addr.sin_addr.s_addr, h->h_addr_list[0], h->h_length);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (connect(sd, (struct sockaddr *)&addr, sizeof(addr)) == -1)
        return -1;

    return sd;
}

static void parse_input(char *line, struct move *m)
{
    static int lookup[] = {7, 6, 5, 4, 3, 2, 1, 0};

    m->frm.y = line[2] - '0';
    m->frm.x = line[5] - '0';

    m->to.y = line[10] - '0';
    m->to.x = line[13] - '0';

    m->frm.y = lookup[m->frm.y];
    m->frm.x = lookup[m->frm.x];
    m->to.y = lookup[m->to.y];
    m->to.x = lookup[m->to.x];
}

static void move_to_json(char *buffer, struct move *m)
{
    static int lookup[] = {7, 6, 5, 4, 3, 2, 1, 0};
    sprintf(buffer, "[[%d, %d], [%d, %d]]",
        lookup[m->frm.y], lookup[m->frm.x], lookup[m->to.y], lookup[m->to.x]);
}

static int find_best_move(AI_instance_t *ai, board_t *board, struct move *m)
{
    int index;

    // generate and find best move
    generate_all_moves(board);
    index = _get_best_move(ai, board);

    memcpy(m, &board->moves[index], sizeof(struct move));

    return index;
}

void usage(char **argv, struct option *options)
{
    int i;
    printf("USAGE: %s <options>\n", argv[0]);
    printf("Available options:\n");
    for (i = 0; options[i].name; i++)
        printf("    -%c, --%s %s\n", options[i].val, options[i].name,
                options[i].has_arg == required_argument ? "<argument>" : "");
}

void parse_arguments(int argc, char **argv)
{
    int c;
    int option_index = 0;
    static struct option long_options[] = {
        {"ai", required_argument, NULL, 'a'},
        {"gui-ip", required_argument, NULL, 'i'},
        {"white", no_argument, NULL, 'w'},
        {"black", no_argument, NULL, 'b'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    while ((c = getopt_long(argc, argv, "a:i:wbvh", long_options,
                    &option_index)) != -1)
        switch (c) {
            case 'a':
                strncpy(ai_file, optarg, sizeof(ai_file));
                break;
            case 'i':
                strncpy(gui_ip, optarg, sizeof(ai_file));
                break;
            case 'w':
                white = 1;
                break;
            case 'b':
                white = 0;
                break;
            case 'v':
                ++verbose;
                break;
            case 'h':
            default:
                usage(argv, long_options);
                exit(0);
        }
}

int main(int argc, char *argv[])
{
    int sd, gui, count = 0;
    char buffer[512];
    board_t *board;
    AI_instance_t *ai;
    struct move m;

    parse_arguments(argc, argv);

    init_magicmoves();

    // connect to GUI
    printf("Connecting..\n");
    if (white) {
        sd = setup_socket(4444);
        gui = wait_for_node(sd);
        close(sd);
        white = 1;
    } else {
        while ((gui = node_connect(gui_ip, 4444)) == -1)
            usleep(1000 * 100);
    }

    // initialize board and AI
    board = new_board(DEFAULT_FEN);
    ai = load_ai(ai_file, 1000);
    if (!ai) {
        printf("failed to load ai\n");
        return 1;
    }

    // get player name and send AI's player name
    memset(buffer, 0, sizeof(buffer));
    if (read(gui, buffer, sizeof(buffer)) <= 0)
        return 1;
    if (write(gui, AI_NAME, strlen(AI_NAME) + 1) <= 0)
        return 1;

    if (verbose > 0)
        print_board(board->board);

    memset(buffer, 0, sizeof(buffer));
    if (white)
        goto this_is_so_dirty_it_is_sexy;

    // game loop
    while (read(gui, buffer, sizeof buffer) > 0) {
        // do remote move
        parse_input(buffer, &m);
        do_actual_move(board, &m);
        swapturn(board);

        if (verbose > 0) {
            printf("%d:\n", ++count);
            print_board(board->board);
        }

this_is_so_dirty_it_is_sexy:
        // find and do AI move
        find_best_move(ai, board, &m);
        do_actual_move(board, &m);
        swapturn(board);

        // send the move to the remote GUI
        move_to_json(buffer, &m);
        if (write(gui, buffer, strlen(buffer)) <= 0)
            break;

        if (verbose > 0)
            print_board(board->board);
    }

    close(gui);

    return 0;
}
