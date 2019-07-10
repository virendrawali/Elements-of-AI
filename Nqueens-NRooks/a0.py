#!/usr/bin/env python3
######################################################################
# This program is to solve nrook, nqueen and nknight problem.
# You need to pass atleast 3 argumets
# 1) Game name(nrook,nqueen or nknight)
# 2) Value of N i.e. number of rooks or queen on solution board
# 3) Number of unavailable position
# 4) If number of unavailable positions greater than 0, then specify
# row and column pair to block
######################################################################
import numpy as np
import sys

######################################################
#Common Functions
######################################################

######################################################
# Function to take sum of rows
######################################################
def row_check(board, row):
    return sum( board[row] )

######################################################
# Function to take sum of column
######################################################
def column_check(board, col):
    return sum( [ row[col] for row in board ] )

######################################################
# Function to display solution board
######################################################
def print_solution_board(input_board,game_name):
    show_board = ""
    for row in range(0,N):
        for column in range(0,N):
            if(input_board[row][column] == 4):
                show_board = show_board + "X" + " "
                continue
            if(input_board[row][column] == 1 ):
                show_board = show_board + game_name + " "
            else:
                show_board = show_board + "_" + " "
        show_board = show_board + "\n"
    show_board.rstrip("\n")
    print(show_board)

######################################################
#Function to create initial board i.e. initial state
######################################################
def create_initial_board(input_list,input_board):
    block_row = []
    block_column = []
    for i in range(1, len(input_list)):
        if i == 1:
            game_name = input_list[i]
        elif i == 2:
            N = int(input_list[i])
        elif i == 3:
            number_of_blocks = int(input_list[i])
            if(number_of_blocks == 0):
                return input_board,game_name
        elif i == 4:
            for j in range(4, len(input_list)):
                if (j % 2 == 1):
                    if ((int(input_list[j])) >= N+1):
                        print("Column Number is greater than size of board,Please enter proper column number")
                        exit()
                    block_column.append(int(input_list[j]) - 1)
                else:
                    if ((int(input_list[j])) >= N+1):
                        print("Row Number is greater than size of board,Please enter proper column number")
                        exit()
                    block_row.append(int(input_list[j]) - 1)
        else:
            break

    for i in range(0, len(block_row)):
        for j in range(0, len(block_column)):
            if (i == j):
                input_board[block_row[i]][block_column[j]] = 4
    return input_board,game_name

######################################################
#Queen Functions
######################################################

######################################################
# Function to mark diagonal places of queen
######################################################
def mark_queen_places(row,column,queen_board,N):
    for ib in range(1,row+1):
        if ((row - ib >= 0) and (column + ib <= N-1) and ((queen_board[row - ib][column + ib] != 4))):
            queen_board[row - ib][column + ib] = 2
        if ((row - ib >= 0) and (column - ib >= 0) and (queen_board[row - ib][column - ib] != 4)):
            queen_board[row - ib][column - ib] = 2
    for ia in range(1, N-row):
        if((row + ia <= (N-1)) and (column + ia <= (N-1)) and (queen_board[row + ia][column + ia] != 4)):
            queen_board[row + ia][column + ia] = 2
        if((row + ia <= (N-1)) and (column - ia >= 0) and ((queen_board[row + ia][column - ia] != 4))):
            queen_board[row + ia][column - ia] = 2
    return queen_board

######################################################
# Function to create successor states of N-Queen board
######################################################
def create_successor_board_queen(board, row, column):
    queen_board = board.copy()
    for r in range(0, N):
        for c in range(0, N):
            if((queen_board[r][c] == 1) or (queen_board[r][c] == 2) or (queen_board[r][c] == 4)):
                continue
            if(r == row and c == column):
                queen_board[r][c] = 1
            else:
                queen_board[r][c] = 0
    queen_board = mark_queen_places(row,column,queen_board,N)
    return queen_board

################################################################
# Function to get list of successor boards of given board
################################################################
def queen_successors(board, Nn):
    state_space = []
    for c in range(0, N):
            #print("Value of Nn",Nn)
            if((column_check(board,c)%2 == 1) or (board[Nn][c] == 2) or (board[Nn][c] == 4)):
                continue
            else:
                temp_board = create_successor_board_queen(board, Nn, c)
                state_space.append(temp_board)
    return state_space
#################################################################
# Function to check all queens are placed on board
#################################################################
def goal_check_queen(check_goal):
    for row in range(0,N):
            if((row_check(check_goal,row)%2 != 1)):
                return 0
    return 1

##########################################################################
# Function to check each state board and decide if solution state is found
##########################################################################
def solve(initial_board):
    fringe = [initial_board]
    if(row_check(initial_board,0)%2!=1):
        count = 0
    else:
        count = 1
    while len(fringe) > 0:
        if(len(fringe)>1):
            count = fringe.pop()
        a = fringe.pop()
        count_flag = 0
        for s in queen_successors(a,count):
            if(count<=N-1 and count_flag == 0):
                count = count +1
                count_flag = 1
            if goal_check_queen(s):
                    return(s)
            fringe.append(s)
            fringe.append(count)
    return []

####################################################
# Knight Functions
###################################################

############################################################
# Function to mark probable places of Knight
############################################################
def mark_knight_places(row,column,knight_board,N):
    if ((row - 2) >= 0 and (column - 1) >= 0 and knight_board[row - 2][column -1]!= 4):
        knight_board[row - 2][column - 1] = 2
    if ((row - 2) >= 0 and (column + 1) <= N-1 and knight_board[row - 2][column + 1]!= 4):
        knight_board[row - 2][column + 1] = 2
    if ((row - 1) >= 0 and (column - 2) >= 0 and knight_board[row - 1][column - 2]!= 4):
        knight_board[row - 1][column - 2] = 2
    if ((row - 1) >= 0 and (column + 2) <= N-1 and knight_board[row - 1][column + 2]!= 4):
        knight_board[row - 1][column + 2] = 2
    if ((row + 1) <= N-1 and (column - 2) >= 0 and knight_board[row + 1][column - 2]!= 4):
        knight_board[row + 1][column - 2] = 2
    if ((row + 1) <= N-1 and (column + 2) <= N-1 and knight_board[row + 1][column + 2]!= 4):
        knight_board[row + 1][column + 2] = 2
    if ((row + 2) <= N-1 and (column - 1) >= 0 and knight_board[row + 2][column - 1]!= 4):
        knight_board[row + 2][column - 1] = 2
    if ((row + 2) <= N-1 and (column + 1) <= N-1 and knight_board[row + 2][column + 1]!= 4):
        knight_board[row + 2][column + 1] = 2
    return knight_board

###################################################################
# Check whether N Knights are placed on board
###################################################################

def goal_check_knight(input_board):
    count = 0
    for row in range(0,N):
        for column in range(0,N):
            if(input_board[row][column] == 1):
                count = count + 1
                if(count == N):
                    return 1

################################################################
# Function to create solution board
################################################################
def create_goal_state_knights(input_board, N):
    for row in range(0,N):
        for column in range(0,N):
            if ((input_board[row][column] == 1) or (input_board[row][column] == 2) or (input_board[row][column] == 4)):
                continue
            else:
                if (goal_check_knight(input_board)):
                    return input_board
                input_board[row][column] = 1
                input_board = mark_knight_places(row,column,input_board,N)
    return input_board

#######################################
# rooks functions
#######################################

################################################################
# Function to create solution board for N-rook
################################################################
def create_goal_state_rooks(input_board, N):
    for row in range(0,N):
        for column in range(0,N):
            if ((input_board[row][column] == 1)):
                break
            elif ((row_check(input_board, row)%2 == 1) or (column_check(input_board, column)%2 == 1) or input_board[row][column] == 4):
                continue
            else:
                input_board[row][column] = 1

######################################################
# Function to create successor states of N-Queen board
######################################################
def create_successor_board_rook_new(board, row, column):
    rook_board = board.copy()
    for r in range(0, N):
        for c in range(0, N):
            if((rook_board[r][c] == 1) or (rook_board[r][c] == 2) or (rook_board[r][c] == 4)):
                continue
            if(r == row and c == column):
                rook_board[r][c] = 1
            else:
                rook_board[r][c] = 0
    return rook_board

################################################################
# Function to get list of successor boards of given board
################################################################
def rooks_successors(board, Nn):
    state_space = []
    for c in range(0, N):
            if((column_check(board,c)%2 == 1) or (board[Nn][c] == 2) or (board[Nn][c] == 4)):
                continue
            else:
                temp_board = create_successor_board_rook_new(board, Nn, c)
                state_space.append(temp_board)
    return state_space
#################################################################
# Function to check all queens are placed on board
#################################################################
def goal_check_rooks(check_goal):
    for row in range(0,N):
            if((row_check(check_goal,row)%2 != 1)):
                return 0
    return 1

##########################################################################
# Function to check each state board and decide if solution state is found
##########################################################################

def solve_rook(initial_board):
    fringe = [initial_board]
    if(row_check(initial_board,0)%2!=1):
        count = 0
    else:
        count = 1
    while len(fringe) > 0:
        if(len(fringe)>1):
            count = fringe.pop()
        a = fringe.pop()
        flag = 0
        for s in rooks_successors(a,count):
            if(count<=N-1 and flag == 0):
                count = count +1
                flag = 1
            if goal_check_rooks(s):
                    return(s)
            fringe.append(s)
            fringe.append(count)
    return []


##############################################################
# Program starts here, here we reading command line arguments 
# deciding and which N-game(nrook,nqueen or nknight) we need
# to solve
##############################################################
N = int(sys.argv[2])
input_list = []
for i in range(0,len(sys.argv)):
        input_list.append(sys.argv[i])
input_board = np.zeros((N,N), dtype=int)
(initial_board,game_name) = create_initial_board(input_list,input_board)
if(game_name == "nrook"):
    solution = solve_rook(input_board)
    if (len(solution) > 0):
        print_solution_board(solution, "R")
    else:
        print("There is no solution for N-Rook game")
elif(game_name == "nqueen"):
    solution = solve(input_board)
    if (len(solution) > 0):
        print_solution_board(solution, "Q")
    else:
        print("There is no solution for N-Queen game")
elif(game_name == "nknight"):
    solution_board = create_goal_state_knights(input_board, N)
    print_solution_board(solution_board, "K")
else:
    print("Please enter nrook, nqueen or nknight")

