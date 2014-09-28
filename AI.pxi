# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 16:15:36 2014

@author: Andreas
"""
import numpy as np
cimport numpy as np
from cppmap import Memory

cdef class AI:
    cdef memory
    cdef np.ndarray piecess
    cdef np.ndarray m 
    cdef np.ndarray features 
    cdef int nr_features
    cdef shortmemory
    cdef int move_nr
    def __init__(self):
        
        cdef int nr_features= 10000
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] piecess =np.array(range(1,8191), dtype=np.uint16)
        cdef np.ndarray[np.int8_t, ndim=1, mode="c"] m = np.zeros((nr_features), dtype=np.int8)
        cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] features = np.random.choice(piecess, (nr_features,8,8)).view('uint16') 
        self.memory = Memory()
        self.piecess = piecess
        self.m = m
        self.features = features
        self.nr_features = nr_features
        self.move_nr = 0
        self.shortmemory = Memory()
    cdef _get_best_move(self, Board board, legal_moves):
        scores = []
        
        for e in legal_moves:
            board.move(e)
            for i in range(self.nr_features):        
                self.m[i] = board.multiply(self.features[i])
            scores.append(self.memory.lookup(self.m.data))
            board.reverse_move()
            
        return  np.argmax(scores)     
        
    cdef do_best_move(self, Board board):        
        legal_moves = board.get_all_legal_moves()
        cdef int best_move = self._get_best_move(board,legal_moves)
        board.move(legal_moves[best_move])
        for i in range(self.nr_features):        
                self.m[i] = board.multiply(self.features[i])
        self.memory.rememberaction(self.m.data)
        
        return legal_moves[best_move] 

    # EVERYTHING YOU DO IS WRONG    
    cdef punish(self):
        self.memory.weaken_axons()
    cdef reward(self):
        self.memory.strengthen_axons()