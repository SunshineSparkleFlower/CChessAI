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

    def print_board(self):
        print_board(self.cboard.board)
