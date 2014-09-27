# distutils: language = c++

cdef extern from "<map>" namespace "std":
    cdef cppclass mymap "std::map<np.ndarray[np.uint16_t, ndim=2], float>":
        mymap()
        float& operator[] (const int& k)

cpdef shits(int nr_of_shits):
    cdef mymap m = mymap()
    cdef int i
    cdef float value

    for i in range(100):
        value = 3.0 * i**2
        m[i] = value

    print m[nr_of_shits]
