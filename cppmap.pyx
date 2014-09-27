# distutils: language = c++

cdef extern from "<map>" namespace "std":
    cdef cppclass mymap "std::map<int, float>":
        mymap()
        float& operator[] (const int& k)

def shits():
    cdef mymap m = mymap()
    cdef int i
    cdef float value

    for i in range(100):
        value = 3.0 * i**2
        m[i] = value

    print m[10]
