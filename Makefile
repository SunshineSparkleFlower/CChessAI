CC=gcc
CXX=g++
CFLAGS=-Wall -shared -fPIC -I/usr/include/python2.7/ -O2
LDFLAGS=

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

all: cython cppmap start

debug: CFLAGS += -DDEBUG -g
debug: all

start: start.o rules.o common.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@.so 

rules: common.o rules.o
	$(CC) $^ -o rules

cppmap: cppmap.o
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@.so 
	
cppmap.o: cppmap.c
	$(CXX) $< $(CFLAGS) -c -o $@

cython:
	cython start.pyx
	cython cppmap.pyx

clean:
	-$(RM) *.o start.so rm cppmap.c start.c
