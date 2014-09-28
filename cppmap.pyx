# distutils: language = c++
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector


cdef class Memory:
    cdef map[string, float] mymap
    cdef vector[string] shortmemory
    cdef float lr_punish
    cdef float lr_reward
    def __init__(self):     
         cdef map[string, float] mymap
         self.mymap = mymap
         cdef vector[string] shortmemory
         self.shortmemory = shortmemory
         self.lr_punish = 0.1
         self.lr_reward = 0.1
         
    cpdef float lookup(self, string key):
        return self.mymap[key]
    
    cpdef float rememberaction(self, string key):
        self.shortmemory.push_back(key)
        

    cdef float reward_func(self, float prew):
        return min(prew * (1 - self.lr_reward) +  1 * (self.lr_reward), 1)
    
    cdef float punish_func(self, float prew):
        return max(prew * (1 - self.lr_punish) -  1 * (self.lr_punish), -1)
        
    cpdef string strengthen_axons(self):
        while self.shortmemory.empty() == False:    
            mem = self.shortmemory.back()    
            self.shortmemory.pop_back()
            self.mymap[mem] = self.reward_func(self.mymap[mem])
            print self.mymap[mem]
    
    cpdef string weaken_axons(self):
        while self.shortmemory.empty() == False:    
            mem = self.shortmemory.back()    
            self.shortmemory.pop_back()
            self.mymap[mem] = self.punish_func(self.mymap[mem])
           # print self.mymap[mem]