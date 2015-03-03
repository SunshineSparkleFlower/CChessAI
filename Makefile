CC=gcc
CFLAGS=-Wall -O2 -funroll-loops -msse4.1 -mavx2 -g
LDFLAGS=-lpthread -lm

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: fast

#fast: CFLAGS += -D UCI_DEBUG
#fast: CFLAGS += -D BOARD_CONSISTENCY_TEST
fast: C_AI.o board.o AI.o common.o bitboard.o magicmoves.o threadpool.o uci.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

alice: CFLAGS += -D ENGINE_NAME=\"Alice\" -D ENGINE_AUTHOR=\"Andreas\"
alice: CFLAGS += -D DISABLE_CASTLING -D DISABLE_EN_PASSANT
alice: alice.o AI.o board.o bitboard.o magicmoves.o common.o uci.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

ai_debug: ai_debug.o board.o AI.o common.o bitboard.o magicmoves.o uci.o

debug: CFLAGS += -DDEBUG -g
debug: test

#test: CFLAGS += -D BOARD_CONSISTENCY_TEST
#test: CFLAGS += -D INSPECT_MOVES
#test: CFLAGS += -D UCI_DEBUG
test: unit_test_ai.o board.o AI.o common.o bitboard.o magicmoves.o uci.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

bench: bench.o board.o AI.o common.o bitboard.o magicmoves.o uci.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

remote: remote.o board.o bitboard.o magicmoves.o AI.o common.o uci.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	-$(RM) *.o C_AI fast test bench ai_debug alice
