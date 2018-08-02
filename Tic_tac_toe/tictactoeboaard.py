import numpy as np


class Player:

    def __init__(self,name,symbol):

        self._name = name
        self._symbol = symbol

    def get_symbol(self):

        return self._symbol

class Board:

    def __init__(self):

        self._board = np.zeros_like([3,3])
        self._last_player = None

    def update_square(self,player, pos):

        self._last_player = player
        self._board[pos] = player.get_symbol()


    def is_winner(self):

        symbol = self._last_player.get_symbol()
        flag = False





