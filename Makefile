CC=gcc
CFLAGS=-O2 -msse4.1 -g
LDFLAGS=-lcfu -lpthread

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: C_AI

C_AI: C_AI.o board.o AI.o map.o rules.o common.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

shits.o: shits.c
	$(CC) $< $(CFLAGS) --std=c99 -c -o $@

fast: CFLAGS += -DFASTRULES
fast: C_AI.o board.o AI.o map.o fastrules.o rules.o common.o shits.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

debug: CFLAGS += -DDEBUG -g
debug: test

test: CFLAGS += -DFASTRULES -g
test: unit_test_ai.o board.o AI.o map.o fastrules.o rules.o common.o shits.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	-$(RM) *.o C_AI fast test
