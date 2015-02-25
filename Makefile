CC=gcc
CFLAGS=-Wall -O2 -funroll-loops -msse4.1 -mavx2 -g
LDFLAGS=-lpthread -lm

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: fast

fast: C_AI.o board.o AI.o common.o bitboard.o magicmoves.o threadpool.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

alice: CFLAGS += -D ENGINE_NAME=\"Alice\" -D ENGINE_AUTHOR=\"Andreas\"
alice: alice.o AI.o board.o bitboard.o magicmoves.o common.o threadpool.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

ai_debug: ai_debug.o board.o AI.o common.o bitboard.o magicmoves.o

debug: CFLAGS += -DDEBUG -g
debug: test

test: unit_test_ai.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

bench: bench.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	-$(RM) *.o C_AI fast test bench ai_debug alice
