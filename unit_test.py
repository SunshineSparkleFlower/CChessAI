#!/usr/bin/python

import start

board = start.Board("r6P/p7/7/4k3/8/8/P7/5KRr w - 0 1")
board.print_board()

board.calculate_legal_moves(0, 6)
print board.get_legal_moves()
