CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-Wall -g -O2 -msse4.1 -L /usr/local/cuda/lib64
LDFLAGS=-lpthread -lm -lrt -lcudart -lcuda

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

%.o: %.cu
	$(NVCC) $< -g -c -o $@

all: fast

fast: C_AI.o board.o AI.o common.o bitboard.o magicmoves.o threadpool.o nand.o
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@

ai_debug: ai_debug.o board.o AI.o common.o bitboard.o magicmoves.o

debug: CFLAGS += -DDEBUG -g
debug: test

test: unit_test_ai.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

bench: bench.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	-$(RM) *.o C_AI fast test bench ai_debug
