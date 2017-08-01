"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from math import sqrt, exp

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    player_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    play_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(game.get_opponent(player))

    advantage_factor = 1                                 
    
    dist = sqrt( (play_loc[0]-opp_loc[0])**2 + (play_loc[1]-opp_loc[1])**2 )
    rel = exp( -dist )
    if opp_moves:    
        rand=random.randint(0, len(opp_moves)-1)
        if sum([opp_moves[rand][0] == 1, opp_moves[rand][0] == 7]) == 1 or sum([opp_moves[rand][1] == 1, opp_moves[rand][1] == 7]) == 1:
            advantage_factor *=2.5 
    own_score = float(len(player_moves)*advantage_factor)
    opp_score = float(len(opp_moves))
    return float(rel*(own_score - opp_score*2))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2*opp_moves)



def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves*3)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            best_move = self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def Max_Value(game, depth): #maximizing player
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth == 0 or len(game.get_legal_moves()) == 0:
                return self.score(game, self)
            v = float("-inf")
            legal_moves = game.get_legal_moves()
            for move in legal_moves: #iterate over all possible moves and return the maximum value
                v = max(v, Min_Value(game.forecast_move(move), depth-1))
            return v
        def Min_Value(game, depth): #minimizing player
            if self.time_left() < self.TIMER_THRESHOLD: #timeout
                raise SearchTimeout()
            if depth == 0 or len(game.get_legal_moves()) == 0:
                return self.score(game, self)
            v = float("inf")
            legal_moves = game.get_legal_moves()
            for move in legal_moves:#iterate over all possible moves and return the minimum value
                v = min(v, Max_Value(game.forecast_move(move), depth-1))
            return v
        legal_moves = game.get_legal_moves() 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if not legal_moves:
            return (-1, -1)
        best_move = legal_moves[random.randint(0, len(legal_moves)-1)] #pick random move to return something
        value = float("-inf")
        v = float("-inf")
        for move in legal_moves: #iterate over possible moves and return move that will lead to maximum value according to heuristic
            v = max(Min_Value(game.forecast_move(move), depth-1), v) #maximizing player
            if v > value:
                value = v
                best_move = move
        return best_move #return move that maximizes heuristic function
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            iterative_depth = 1
            while True: #loop until error occurs
                best_move = self.alphabeta(game, iterative_depth) #call alphabeta method with current depth
                iterative_depth+=1
        except SearchTimeout:
            pass
        return best_move
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def max_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth == 0 or len(game.get_legal_moves()) == 0: #if no more leaves in the tree or more than depth specified return heuristic
                return self.score(game, self)
            v = float("-inf") #assign -infinity as minimal value
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                next_game = game.forecast_move(move)
                v = max(v, min_value(next_game, depth-1, alpha, beta))
                if v >= beta: #prune if value larger or equal to beta since maximizing player will choose this value or greater
                    return v
                alpha = max(v, alpha) #keep track of alpha for pruning at the next level down (minimizing player)
            return v
        def min_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth == 0 or len(game.get_legal_moves()) == 0:
                return self.score(game, self)
            v = float("inf")
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                next_game = game.forecast_move(move)
                v = min(v, max_value(next_game, depth-1, alpha, beta))
                if v <= alpha: #prune
                    return v
                beta = min(v, beta)
            return v
        #instantiation of process
        if self.time_left() < self.TIMER_THRESHOLD: 
            raise SearchTimeout()
        legal_moves = game.get_legal_moves() 
        if not legal_moves:
            return (-1, -1)
        best_move = legal_moves[random.randint(0, len(legal_moves)-1)] #pick random move to return something
        score = float("-inf")
        for move in legal_moves:
            next_game = game.forecast_move(move)
            v = min_value(next_game, depth-1, alpha, beta)
            if score < v:#keep track of best score & move
                score = v
                best_move = move
                if v>=beta: #leads to win
                    return best_move
            alpha = max(v, alpha)
        return best_move
