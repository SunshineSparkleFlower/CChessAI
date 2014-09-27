import numpy as np
cimport numpy as np

cdef class Board:
    cdef board_t cboard
    cdef np.ndarray npboard 

    def __init__(self, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] board = np.zeros((8,8), dtype=np.uint16)

        board.fill(P_EMPTY)
        self.npboard = board

        fen_parts = fen.split(' ')
        fen = fen_parts[0].split('/')

        # initialize board
        row = 7
        for rank in fen:
            col = 0
            for c in rank:
                if c.isdigit():
                    col += int(c)
                else:
                    board[row][col] = self._fen_to_chesspiece(c)
                    col += 1
            row -= 1

        self.cboard.board = <piece_t *>board.data
        self.cboard.moves_count = 0
        self.cboard.turn = WHITE if fen_parts[1] == 'w' else BLACK
    
    cpdef multiply(self, np.ndarray[np.uint16_t, ndim=2] f):
        print  self.npboard
        return  np.array_equal(np.bitwise_and(self.npboard,f), f)
        
    def _fen_to_chesspiece(self, c):
        to_piece = {
            'P': WHITE_PAWN,
            'N': WHITE_KNIGHT,
            'B': WHITE_BISHOP,
            'R': WHITE_ROOK,
            'Q': WHITE_QUEEN,
            'K': WHITE_KING,
            'p': BLACK_PAWN,
            'n': BLACK_KNIGHT,
            'b': BLACK_BISHOP,
            'r': BLACK_ROOK,
            'q': BLACK_QUEEN,
            'k': BLACK_KING,
        }
        return to_piece[c]

    def do_move(self, fromy, fromx, toy, tox):
        cdef piece_t backup

        cdef coord_t frm
        frm.y = fromy
        frm.x = fromx
        
        cdef coord_t to
        to.y = toy
        to.x = tox
        
        do_move(self.cboard.board, frm, to, &backup)
        
    def print_board(self):
        print_board(self.cboard.board)

    def calculate_legal_moves(self, row, col):
        cdef coord_t coord

        coord.y = row
        coord.x = col
        get_legal_moves(&self.cboard, &coord)

    def print_legal_moves(self):
        print "number of legal moves: ", self.cboard.moves_count
        print "Legal moves:"
        for i in xrange(0, self.cboard.moves_count):
            print self.cboard.moves[i]

    def get_legal_moves(self):
        ret = []
        for i in xrange(0, self.cboard.moves_count):
            ret += [self.cboard.moves[i]]

        return ret
