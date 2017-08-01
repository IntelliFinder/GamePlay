"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

from isolation import Board
from game_agent import MinimaxPlayer, AlphaBetaPlayer
from sample_players import GreedyPlayer, RandomPlayer

import game_agent

from importlib import reload

class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)
        
    def testBoard(self):
        AB     = AlphaBetaPlayer()
        MM     = MinimaxPlayer()
        greedy = GreedyPlayer()
        rand   = RandomPlayer()
        game   = Board(AB, MM)
        
        game.apply_move((1,5))
        game.applymove((2,3))
        
        self.assertTrue( game.check_legal_move((4,5)) )
        self.assertEqual( (2,3), game.get_player_location(game._active_player) )
        
if __name__ == '__main__':
    unittest.main()
