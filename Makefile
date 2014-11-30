CC=gcc
CXX=g++
CFLAGS=-Wall -g -O2 -msse4.1 -L /usr/local/cuda/lib64
LDFLAGS=-lpthread -lm -lrt -lcudart -lcuda

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: fast

fast: C_AI.o board.o AI.o common.o bitboard.o magicmoves.o threadpool.o nand.o
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@

ai_debug: ai_debug.o board.o AI.o common.o bitboard.o magicmoves.o

nand.o: nand.cu
	nvcc $< -c -o $@
AI.o: AI.cu
	nvcc $< -c -o $@
debug: CFLAGS += -DDEBUG -g
debug: test

test: unit_test_ai.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

bench: bench.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	-$(RM) *.o C_AI fast test bench ai_debug