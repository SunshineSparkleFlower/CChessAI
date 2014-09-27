CC=gcc
CFLAGS=-Wall -shared -fPIC -I/usr/include/python2.7/ -O2
LDFLAGS=

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: cython start

test: cython start unit_test

unit_test: unit_test.o rules.o common.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@.so 

debug: CFLAGS += -DDEBUG -g
debug: all

start: start.o rules.o common.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@.so 

rules: common.o rules.o
	$(CC) $^ -o rules

cython:
	cython start.pyx
	cython unit_test.pyx

clean:
	-$(RM) *.o start.so
