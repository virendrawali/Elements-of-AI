#!/usr/bin/env python3
from operator import itemgetter
from itertools import groupby
import sys


MAX_DEPTH = 3 # Maximum depth for minimax algorithm with aplha-beta pruning


class BetsyBoard:
    def __init__(self, n, board=None, move=None):
        '''
            Represents the state of the Betsy game board
        '''
        self.n = n
        self.board = board if board else '.' * (n*(n+3))
        self.move = move

    def drop_pebble(self, column_id, color):
        column = self.board[column_id::self.n]
        position = column.rfind('.')

        assert position != -1, 'column should not be full'

        n = self.n
        board = ''.join(color if c == column_id and r == position else self.board[n*r+c]
                        for r in range(n+3) for c in range(n))
        return BetsyBoard(n, board, move=column_id+1)


    def rotate_column(self, column_id):
        column = self.board[column_id::self.n]

        assert any(each != '.' for each in column), 'column should not be empty'

        bottom_element = column[-1]
        position = column.rfind('.')
        column = column[-1] + column[:-1] if position == -1 else column[:position+1] + bottom_element + column[position+1:-1]

        n = self.n
        board = ''.join(column[r] if c == column_id else self.board[n*r+c] for r in range(n+3) for c in range(n))
        return BetsyBoard(n, board, move=-(column_id+1))


    def get_winner(self):
        '''
            Returns:
                'x' if 'x' won, 'o' if 'o' won and None otherwise
                if both x and o have completed sequences, then it returns '-'
        '''
        n = self.n
        square = self.board[:n*n]


        left_diagonal = ''.join(square[n*i + i] for i in range(n))
        right_diagonal = ''.join(square[n*i + n-i-1] for i in range(n))

        diagonals = [left_diagonal, right_diagonal]
        columns = [square[c::n] for c in range(n)]
        rows = [square[n*i: n*i+n] for i in range(n)]

        sequences = rows + columns + diagonals

        winners = set()
        for sequence in sequences:
            first = sequence[0]
            if first == '.':
                # if the first item is empty
                # then this isn't a winning sequence
                continue
            if any(item != first for item in sequence):
                # if any item is not same as first item in sequence
                # then this isn't a winning sequence
                continue

            # all items are same as first item and first item is not empty
            # so, this is a winning sequence
            winners.add(first)

        if len(winners) == 0:
            return None
        elif len(winners) == 1:
            return list(winners)[0]
        else:
            return '-' # both x and o have completed sequences

    def is_full(self):
        return not ('.' in self.board)

    def __repr__(self):
        n = self.n
        return '\n'.join(' '.join(self.board[n*i: n*(i+1)]) for i in range(n+3))


def successors(state, color):
    '''
        Generates successor boards for a given board and the player with the current turn

        Returns:
            list of BetsyBoard objects
    '''
    order = list(range(state.n))
    assert(len(order) == state.n)
    for column_id in order:
        try:
            s = state.rotate_column(column_id)
            if s.board != state.board:
                yield s
        except AssertionError:
            continue

    n = state.n
    if state.board.count(color) <= n*(n+3)/2:
        for column_id in order:
            try:
                yield state.drop_pebble(column_id, color)
            except AssertionError:
                continue



def evaluate(state, color):
    '''
        Evaluation function definition to calculate the cost of given board
        so that player will decide next potential move

        Parameters:
            state (BetsyBoard)
            color (str): 'x' or 'o'

        Returns
            float, a score for the given state
    '''
    n = state.n
    square = state.board[:n*n]


    # Get the elements of left and right diagonal of the board
    left_diagonal = ''.join(square[n*i + i] for i in range(n))
    right_diagonal = ''.join(square[n*i + n-i-1] for i in range(n))

    diagonals = [left_diagonal, right_diagonal]
    columns = [square[c::n] for c in range(n)]
    rows = [square[n*i: n*i+n] for i in range(n)]

    sequences = rows + columns + diagonals

    # sequence_lengths is a dictionary with length of continuous pieces 'x' or 'o'
    # For n=3, we can have 'x', 'xx' or 'xxx'. Therefore, below dictionary looks like
    # {'x': {1: 0, 2: 0, 3: 0}, 'o': {1: 0, 2: 0, 3: 0}}
    sequence_lengths = {
        'x': { length: 0 for length in range(1, n+1) },
        'o': { length: 0 for length in range(1, n+1) }
    }

    for sequence in sequences:
        subsequences = [(label, sum(1 for _ in group)) for label, group in groupby(sequence)]
        for c, length in subsequences:

            if c == '.':
                continue
            sequence_lengths[c][length] += 1

    # refer attached pdf for detailed explanation
    score = {
        'x': sum(count * (2 ** length) for length, count in sequence_lengths['x'].items()),
        'o': sum(count * (2 ** length) for length, count in sequence_lengths['o'].items())
    }
    return (score[color] - score[switch_turn(color)] )/ (2**n)




def max_value(state, color, depth, alpha, beta, max_color):
    '''
        max_value function from minimax algorithm

        Returns:
            float, the alpha value
    '''
    winner = state.get_winner()
    if winner:
        return 1/depth if winner == max_color or winner == '-' else -1/depth

   # if state.is_full():
   #     return 0

    if depth == MAX_DEPTH:
        return evaluate(state, color) / depth

    for s in successors(state, color):
        alpha = max(alpha, min_value(s, switch_turn(color), depth+1, alpha, beta, max_color))
        if alpha >= beta:
            return alpha

    return alpha


def min_value(state, color, depth, alpha, beta, max_color):
    '''
        min_value function from minimax algorithm

        Returns:
            float, the alpha value
    '''
    winner = state.get_winner()
    if winner:
        # return positive value if max wins and negative when min wins
        return 1/depth if winner == max_color else -1/depth

  # if state.is_full():
  #     return 0

    if depth == MAX_DEPTH:
        return -evaluate(state, color)/depth

    for s in successors(state, color):
        beta = min(beta, max_value(s, switch_turn(color), depth+1, alpha, beta, max_color))
        if alpha >= beta:
            return beta

    return beta


def minimax_decision(state, color, max_color):
    scores_with_states = [(min_value(s, switch_turn(color), depth=1, alpha=float('-inf'), beta=float('inf'), max_color=max_color), s)
                        for s in successors(state, color) if s.board != state.board]

    max_score, max_state = max(scores_with_states, key=itemgetter(0))
    return max_state


def switch_turn(color):
    return 'x' if color == 'o' else 'o'



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: ./betsy.py [n] [max_color] [board_string] [time - optional]')
        exit()

    n = int(sys.argv[1])
    max_color = sys.argv[2]
    board_string = sys.argv[3]

    for depth in [3,7,8,9,10]:
        MAX_DEPTH =  depth
        #Instantiating the board class with input board
        board = BetsyBoard(n=n, board=board_string)
        #print (board, '\n')
        board = minimax_decision(board, max_color, max_color)
        #print (board, '\n')
        if(board.move < 0):
            print("I'd recommend rotating column {}".format(-board.move))
        else:
            print("I'd recommend dropping a pebble in column {}".format(board.move))
        print ('%s %s' % (board.move, board.board))
