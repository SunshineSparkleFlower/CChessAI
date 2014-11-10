CC=gcc
CXX=g++
CFLAGS=-Wall -g -O2 -msse4.1 -L /usr/local/cuda/lib64
LDFLAGS=-lpthread -lm -lrt -lcudart -lcuda

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: fast

fast: nand.o board.o AI.o common.o bitboard.o magicmoves.o
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@

ai_debug: ai_debug.o board.o AI.o common.o bitboard.o magicmoves.o

debug: CFLAGS += -DDEBUG -g
debug: all

nand.o: nand.cu
	nvcc $< -c -o $@

clean:
	-$(RM) *.o C_AI fast test bench ai_debug
