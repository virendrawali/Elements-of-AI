import pprint
import copy
import numpy as np
from collections import namedtuple
import types

alphabeta_depth = 1

def count_row(board,turn):
    cost = 0
    piece = 0
    n = len(board[0])
    row_list_count = []
    if (turn == "x"):
        opponent = "o"
    else:
        opponent = "x"
    for i in range(0,n):
        players = others = 0
        for j in range(0,n):
            if(turn == board[i][j]):
                players = players + 1
            elif (opponent == board[i][j]):
                others = others + 1
        row_list_count.append((players,others))
        cost = cost + (players*players) - (others*others)
    #print("row list is",row_list_count)
    #print("row cost is",cost)
    return cost


def count_column(board,turn):
    cost = 0
    piece = 0
    column_list_count = []
    n = len(board[0])
    if (turn == "x"):
        opponent = "o"
    else:
        opponent = "x"
    for i in range(0,n):
        players = others = 0
        for j in range(0,n):
            #print("j:",j)
            if(turn == board[j][i]):
                players = players + 1
            elif (opponent == board[j][i]):
                others = others + 1
        column_list_count.append((players,others))
        cost = cost + (players*players) - (others*others)
    #print("column list is", column_list_count)
    #print("column cost:", cost)
    return cost


def count_diagonal(board,turn):
    cost = 0
    piece = 0
    diagonal_list_count = []
    n = len(board[0])
    row = 0
    if (turn == "x"):
        opponent = "o"
    else:
        opponent = "x"
    players = others = 0
    for i in range(0,n):
        if (turn == board[i][i]):
            players = players + 1
        elif (opponent == board[i][i]):
            others = others + 1
    cost = cost + (players * players) - (others * others)
    diagonal_list_count.append((players, others))
    players = others = 0
    for i in range(n-1,-1,-1):
        if (turn == board[row][i]):
            players = players + 1
        elif (opponent == board[row][i]):
            others = others + 1
        row = row + 1
    cost = cost + (players * players) - (others * others)
    diagonal_list_count.append((players, others))
    #print("diagonal",diagonal_list_count)
    #print("diagonal cost:",cost)
    return cost


def heuristic(board,turn):
    #row_cost = count_row(board,turn)
    #column_cost= count_column(board,turn)
    #diagonal_cost = count_diagonal(board,turn)
    #total_cost = row_cost + column_cost + diagonal_cost
    #print("Heuristic value: ", total_cost)
    #print("turn: ", turn)
    #return(total_cost)
    return 0


def print_board(board, space =''):
    [print(space, row) for row in board.state]

#Takes both list of boards and single boards as arguments
def print_tuple(board_data, depth=0):
    space=""
    #If a list of boards is passed, print each of them one by one
    if(isinstance(board_data, list)):
        for board_d in board_data:
            print("After {}'s turn".format(turns[board_d.turn]))
            print("Board: ")
            print_board(board_d,space)
            print("\n")
    #Else print the single board passed as an argument.
    else:
        space = " "*(alphabeta_depth - depth)*board_data.n*3
        print("After {}'s turn".format(turns[board_data.turn]))
        print("Board: ")
        print_board(board_data,space)
        print("\n")


#Function checks for a given board state if there is diagonal considering
#whether it is 'o' or 'x' turn. It only checks the top n rows for the diagonal
def check_diagonal(board, turn=False):

    if(turn == False):
        turn = turns[board.turn]
    r = 0
    left_diag = 0
    while(r < board.n and board.state[r][r] == turn):
        left_diag += 1
        r += 1
    r = 0
    right_diag = 0
    while(r < board.n and board.state[r][board.n -1 -r] == turn):
        right_diag += 1
        r += 1

    if(right_diag == board.n or left_diag == turn):
        return True
    else:
        return False


#Check if the top 3 cols or rows are complete and return True or False
def check_row_col(board):

    turn = turns[board.turn]
    col_counts = [0]*board.n
    #print(col_counts)
    for index in range(0,board.n):
        row_count = 0
        for key, element in enumerate(board.state[index]):
            if(element == turn):
                col_counts[key] += 1
                row_count += 1
        if(row_count == board.n):
            return True
    if(board.n in col_counts):
        print("true")
        return True
    return False


def is_goal_state(board):
    return(check_row_col(board) or check_diagonal(board))


def rotate_pebble(board_data,index):
    #change it
    temp_board_data = namedtuple('Board', ['state', 'n', 'turn'])
    temp_board_data.turn = turns[board_data.turn]
    temp_board_data.n = board_data.n

    temp_board = copy.deepcopy(np.array(board_data.state))
    last_pebble = board_data.state[len(board_data.state)-1][index]
    #pprint.pprint(temp_board.tolist())

    #print("Inside rotate pebble")
    #print("last pebble :",last_pebble)
    for i in range(len(board)-1,-1,-1):
        if (i == 0 and (temp_board[i][index] != ".")):
            temp_board[i][index] = last_pebble
            #pprint.pprint(temp_board.tolist())
            temp_board_data.state = temp_board.tolist()
            return temp_board_data
        if(temp_board[i-1][index] == "."):
            temp_board[i][index] = last_pebble
            #pprint.pprint(temp_board.tolist())
            temp_board_data.state = temp_board.tolist()
            return temp_board_data
        temp_board[i][index] = temp_board[i-1][index]
    return temp_board_data



def drop_pebble(board_data,index,turn):
    temp_board = copy.deepcopy(board_data.state)

    #print("Inside drop pebble")
    #pprint.pprint(temp_board)

    for i in range(len(board_data.state)-1,-1,-1):
        if(temp_board[i][index] == "."):
            temp_board[i][index] = turn
            #pprint.pprint(temp_board)
            temp_board_data = Board(temp_board, board_data.n, turns[board_data.turn])
            return temp_board_data
        else:
            continue


def input_board(input_string):
    str_len = len(input_string)
    row = int(str_len / 3)
    column = 3
    k = 0
    input_board = [[0 for i in range(0, column)] for i in range(0, row)]
    for i in range(0, row):
        for j in range(0, column):
            input_board[i][j] = input_string[k]
            k = k + 1
    return input_board


# Returns 0 for empty, and number of elements for not empty
def check_col_empty(board, col_index):
    count = 0
    for row in board:
        if(row[col_index] == "x" or row[col_index] == "o"):
            count += 1
    return count


def successor(board_data):
    board = board_data.state
    successor_states = []
    i=2
    #print([row[1] for row in board])
    for i in range(0,board_data.n):
        if(check_col_empty(board,i) == (board_data.n+3)):            #If the col is full, we can only rotate it
            successor_states += [rotate_pebble(board_data,i)]
        elif(check_col_empty(board,i) < 2):                         #If the col is empty we can only add a piece
            successor_states += [drop_pebble(board_data,i,board_data.turn)]
        else:
            successor_states += [rotate_pebble(board_data,i)]
            successor_states += [drop_pebble(board_data,i,board_data.turn)]
    #print("Succesor completed")
    #print("length of successor states: ", len(successor_states))
    #print(successor_states)
    return successor_states


def alphabeta(board_data, depth, alpha, beta, maximizingPlayer):
    #node = board_data.state
    print("Alpha_beta - maxPl: {}, alpha: {}, beta: {}, depth: {}".format(maximizingPlayer, alpha, beta, depth))
    print_tuple(board_data, depth)

    if(is_goal_state(board_data)):
        if (turns[board_data.turn] ==  maxPL):
            print("Value: 10")
            return 10
        else:
            print("Value: -10")
            return -10

    if(depth == 0 or is_goal_state(board_data)):
        #print_tuple(board_data)
        print("Depth 0 ")
        return heuristic(board_data.state, turns[board_data.turn])
    if(maximizingPlayer):
        value = -10000
        #for each child of node do
        for child in successor(board_data):
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if(alpha>= beta):
                print("Loop discontinued Alpha,beta:  ", alpha, beta)
                break #(* beta cut-off *)
        return value
    else:
        value = 10000
        for child in successor(board_data):
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if(alpha >=beta):
                print("Loop discontinued Alpha,beta:  ", alpha, beta)
                break
        return value


def solve(initial_board):
    fringe = list()
    fringe.append(initial_board)

    while len(fringe) > 0:
        next_board = fringe.pop(0)
        #print("Creating successors of following board")
        #print_tuple(next_board)
        for s in successor(next_board):
            #print_tuple(s)
            fringe.append(s)



#input_string.split("\n").join('')
# Use this for all the boards
# state - saves the 2D array of current board state_array
# n - saves the input argument n, where n - num of col
# turn - saves whose turn it is which will be required by all functions
Board = namedtuple('Board', ['state', 'n', 'turn'])

#input_string = "...x.oo.ox.oxxxoxo"
#input_string = "ox.oxo.xx.oxxxoxo"
#input_string = "...............oxo"
#input_string = ".................."
input_string = ".....oxxooooxxxooo"
maxPL = 'x'

turns = {'o':'x', 'x':'o'}
board = input_board(input_string)
board_data = Board(board,3,'x')

print(".....................Input board..................")
print_tuple(board_data)
print(".....................................................")

print("Does board has a diagonal: ", check_diagonal(board_data))
print("Is it a goal state: ", is_goal_state(board_data))
values=[]
states=[]
value = -10000


for s in successor(board_data):
    #print("Solving for following min node: ")
    #print_tuple(s)
    value = alphabeta(s, alphabeta_depth, value, 10000, False)
    values.append(value)
    states.append(s)
    #print("Score of the following move: ", value)
    #print_tuple(s)

print(values)
print_tuple(states)
