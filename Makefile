CC=gcc
CFLAGS=-Wall -O2 -msse4.1 -g
LDFLAGS=-lcfu -lpthread

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: C_AI

C_AI: C_AI.o board.o AI.o map.o rules.o common.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

debug: CFLAGS += -DDEBUG -g
debug: all

clean:
	-$(RM) *.o C_AI
