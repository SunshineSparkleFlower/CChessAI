CC=gcc
CFLAGS=-O2 -msse4.1 -g
LDFLAGS=-lpthread -lm

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: fast

fast: C_AI.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

debug: CFLAGS += -DDEBUG -g
debug: test

test: unit_test_ai.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	-$(RM) *.o C_AI fast test
