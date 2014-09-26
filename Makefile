CC=gcc
CFLAGS=-Wall -shared -fPIC -I/usr/include/python2.7/ -O2 -g
LDFLAGS=

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: cython start

start: start.o rules.o common.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@.so 

rules: common.o rules.o
	$(CC) $^ -o rules

cython:
	cython start.pyx

clean:
	-$(RM) *.o start.so
