CC=gcc
CFLAGS=-shared -fPIC -I/usr/include/python2.7/ -O2
LDFLAGS=

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: cython start

start: start.o rules.o common.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@.so 

cython:
	cython start.pyx

clean:
	-$(RM) *.o start.so
