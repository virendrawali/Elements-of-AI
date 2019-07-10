#!/usr/bin/env python3

## ----------- Please refer the Readme.txt file for details -----------------##
##############################################################################
# The program is to solve 16 puzzle problem i.e. scrambled board of 1 to 16 such
#  a way that it will arrange the numbers in 1 to 16 in order in minimum path.
##############################################################################

import sys
import numpy as np
import queue
import time

pri_queue = queue.PriorityQueue()

goal_board = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
goal_index = { goal_board[i][j]: (i, j) for i in range(0,4) for j in range(0,4) }
goal_hash = {1:1,2:2,3:3,4:4,5:1,6:2,7:3,8:4,9:1,10:2,11:3,12:4,13:1,14:2,15:3,16:4}

##############################################################################
# Definition to calculate Manhattan distance of current board with goal board
#  in both direction and taking minimum distance of both
##############################################################################

def manhattan_distance_rollover(initial, final):
    x1, y1 = initial
    x2, y2 = final

    x_distance = min(abs(x1 - x2), 4 - abs(x1 - x2))
    y_distance = min(abs(y1 - y2), 4 - abs(y1 - y2))

    return x_distance + y_distance

##############################################################################
# Definition to take sum of all manhattan distances of each number of current
# board with goal board and dividing by 4 , as we are moving 4 pieces of each
# row in one move. It is used by heuristic2 only
##############################################################################

def heuristic1(state):
    return sum(manhattan_distance_rollover([i, j], goal_index.get(e))
               for i, row in enumerate(state) for j, e in enumerate(row))/4


###############################################################################
# Definition to take reverse permutation of each row-wise and taking maximum of
# this heuristic and heuristic1(Manhattan distance divided by 4)
###############################################################################

def heuristic2(state):
    input_board_new = []
    new_list = []
    row_cost = []
    for i in range(0, N):
        input_board_new.append([])
        for j in range(0, N):
            input_board_new[i].append(goal_hash[state[i][j]])
    for row in input_board_new:
        new_list.append(row[row.index(min(row)):] + row[0: row.index(min(row))])
    for row in new_list:
        count = 0
        for i in range(0,N):
            for j in range(i+1,N):
                if(row[i]>=row[j]):
                 count = count + 1
        row_cost.append(count)
    return max(heuristic1(state),sum(row_cost))

##############################################################################
# Definition to take sum of all manhattan distances of each number of current
# board with goal board and divide the sum by 2
##############################################################################

def heuristic3(state):
    return sum(manhattan_distance_rollover([i, j], goal_index.get(e))
               for i, row in enumerate(state) for j, e in enumerate(row))/2

##############################################################################
# Definition to take sum of all manhattan distances of each number of current
# board with goal board. 
##############################################################################

def heuristic4(state):
    return sum(manhattan_distance_rollover([i, j], goal_index.get(e))
               for i, row in enumerate(state) for j, e in enumerate(row))

############################################################################
# Definition to move each row of a board right by 1 or left by 1
############################################################################

def shift_row(input_board,r,dir):
    state_array = []
    i = 0
    if(dir == 1):
        for row in input_board:
            row_line = []
            if (i == r):
                num = row[3]
                row_line.append(int(num))
                for num1 in range(0, 3):
                    row_line.append(int(row[num1]))
                state_array.append(row_line)
            else:
                for num2 in range(0, 4):
                    row_line.append(row[num2])
                state_array.append(row_line)
            i = i + 1
    if(dir == -1):
        i = 0
        for row in input_board:
            row_line = []
            if (i == r):
                for num1 in range(1, 4):
                    row_line.append(int(row[num1]))
                row_line.append(row[0])
                state_array.append(row_line)
            else:
                for num2 in range(0, 4):
                    row_line.append(row[num2])
                state_array.append(row_line)
            i = i + 1
    return (state_array,("L" if dir == -1 else "R") + str(r + 1))

############################################################################
# Definition to move each column of a board right by 1 or left by 1
############################################################################

def shift_column(input_board,c,dir):
    state_array = []
    temp_board = np.array(input_board)
    temp_board = temp_board.transpose()
    i = 0
    if(dir == 1):
        for row in temp_board:
            row_line = []
            if (i == c):
                num = row[3]
                row_line.append(int(num))
                for num1 in range(0, 3):
                    row_line.append(int(row[num1]))
                state_array.append(row_line)
            else:
                for num2 in range(0, 4):
                    row_line.append(row[num2])
                state_array.append(row_line)
            i = i + 1
    if(dir == -1):
        i = 0
        for row in temp_board:
            row_line = []
            if (i == c):
                for num1 in range(1, 4):
                    row_line.append(int(row[num1]))
                row_line.append(row[0])
                state_array.append(row_line)
            else:
                for num2 in range(0, 4):
                    row_line.append(row[num2])
                state_array.append(row_line)
            i = i + 1
    temp_state_array = np.array(state_array)
    state_array = temp_state_array.transpose()
    return (state_array.tolist(),("U" if dir == -1 else "D") + str(c + 1))

############################################################################
# Definition to check if current state is goal state or not
############################################################################

def is_goal(state):
    N = len(state)
    for i in range(0, N):
        for j in range(0, N):
            if state[i][j] != goal_board[i][j]:
                return False
    return True

############################################################################
# Definition to successor boards for current board
############################################################################

def successors(board):
    state_space = []
    for r in range(0, N):
        for d in (1, -1):
            state_space.append(shift_row(board, r, d))
    for c in range(0, N):
        for d in (1, -1):
            state_space.append(shift_column(board, c, d))
    return state_space

############################################################################
# Definition to solve a board with inadmissible heuristic with can solve any
# board but the path would not be optimal(heuristic4)
############################################################################

def solve_backup_v2(initial_board):
    fringe = queue.PriorityQueue()
    fringe.put((heuristic4(initial_board), 0, initial_board,""))
    open = {}
    closed = set()
    while not fringe.empty():
        f, g, state, route_so_far = fringe.get()
        if is_goal(state):
            print("Final board backup solve:", state)
            print(route_so_far)
            return True
        closed.add(str(state))
        g = g + 1
        for (s,route) in successors(state):
            if str(s) in closed:
                continue
            if str(s) in open:
                if g + heuristic4(s) > open[str(s)]:
                    continue

            open[str(s)] = g + heuristic4(s)
            fringe.put((g + heuristic4(s), g, s, route_so_far + " " + route))
    return False

############################################################################
# Definition to solve a board with inadmissible heuristic(heuristic3)
############################################################################

def solve_backup(initial_board):
    fringe = queue.PriorityQueue()
    fringe.put((heuristic3(initial_board), 0, initial_board,""))
    open = {}
    closed = set()
    solve_start_time = time.time()
    solve_iteration_time = time.time()
    while not fringe.empty():
        f, g, state, route_so_far = fringe.get()
        if (solve_iteration_time - solve_start_time > 900):
            solve_backup_v2(initial_board)
            exit()
        if is_goal(state):
            print("Final board backup solve:", state)
            print(route_so_far)
            return True
        closed.add(str(state))
        g = g + 1
        for (s,route) in successors(state):
            if str(s) in closed:
                continue
            if str(s) in open:
                if g + heuristic3(s) > open[str(s)]:
                    continue

            open[str(s)] = g + heuristic3(s)
            fringe.put((g + heuristic3(s), g, s, route_so_far + " " + route))
            solve_iteration_time = time.time()
    return False

############################################################################
# Definition to solve a board with admissible heuristic(heuristic2)
############################################################################

def solve(initial_board):
    fringe = queue.PriorityQueue()
    fringe.put((heuristic2(initial_board), 0, initial_board,""))
    open = {}
    closed = set()
    solve_start_time = time.time()
    solve_iteration_time = time.time()
    while not fringe.empty():
        f, g, state, route_so_far = fringe.get()
        if(solve_iteration_time - solve_start_time > 420):
            print("Switching to another heuristic function")
            solve_backup(initial_board)
            exit()
        if is_goal(state):
            print("Final board in solve:", state)
            print(route_so_far)
            return True
        closed.add(str(state))
        g = g + 1
        for (s,route) in successors(state):
            if str(s) in closed:
                continue
            if str(s) in open:
                if g + heuristic2(s) > open[str(s)]:
                    continue
            open[str(s)] = g + heuristic2(s)
            fringe.put((g + heuristic2(s), g, s, route_so_far + " " + route))
            solve_iteration_time = time.time()
    return False

############################################################################
# Definition to create a input board from given input file
############################################################################

def get_input_board(input_file):
    file = open(input_file,"r")
    row = 0
    column = 0
    for line in file:
        column = 0
        for i in line.split():
            input_board[row][column] = int(i)
            column = column + 1
        row = row +1

#######################
# Program starts here
#######################

N = 4
input_board = np.zeros((N,N), dtype=int)
get_input_board(sys.argv[1])
print("Array is\n", input_board)
input_board = input_board.tolist()
solve(input_board)
