The program is to solve 16 puzzle problem, i.e. scrambled board of 1 to 16 such a way that it will arrange the numbers from 1 to 16 in order using the minimum path. 

We tried different heuristics to solve the problem. 
1. Some can solve all boards, but the heuristic is not admissible, sometimes it overestimates the actual cost to the goal.
2. Some heuristics are admissible and can solve the boards but take longer time for boards which need a large number of moves to solve.
Therefore, our strategy is to solve the input board with admissible heuristic,
If the heuristic is not able to solve board in 7 minutes then we are switching to inadmissible heuristic to solve the board. If even this heuristic fails to solve in 15 minutes, we will be switching to another inadmissible heuristic which will give answer most of the times but would not be optimal.

1) Heuristic1:
In this heuristic, we are using a slightly modified version of inverse permutation heuristic for each row and another heuristic which is a modified version of Manhattan distance. We are taking the maximum of these two admissible heuristics to compute the heuristic value from the current state.
For example, 

First admissible heuristic:  Calculating modified inverse permutation.
Input Board:
5 7 8 1
10 2 4 3
6 9 11 12
15 13 14 16

1. Identify the smallest number in each row.
2. Rotate the row according to the smallest number and obtain a rotated array like below
 [5 7 8 1] ->> [1 5 7 8]
 [10 2 4 3] ->> [2 4 3 10]
3. For each row, we are checking what should be the actual column position of each number in goal board.  For row  [5 7 8 1], 5 should be present in column 1, 7 in column 3, 8 in column 4 and 1 in column 1 with respect to the goal board. Using this logic below will be the converted board, and on that, we compute the inverse permutation specific to each row.
[1 5 7 8]  ->> [1 1 3 4] 
4. Calculate inverse permutation cost for each number in the row by adding 1 when there is a number greater than or equal to that number in the array.
Inverse permutation cost of [1 1 3 4]: 1 (Because 1 and 1 are equal)
5. We perform the above operations for each row of the board to compute the final value.

Second admissible Heuristic:  Rollover Manhattan distance divided by 4 as we are moving 4 numbers in each row or column move. 

Then we take the max of the sum of each row cost and rollover Manhattan distance.

2) Heuristic2:
In this heuristic, we are dividing the sum of rollover Manhattan distance by 2 which is not admissible for a subset of the states.

3) Heuristic3:
In this heuristic, we are taking the rollover Manhattan distance of each number in given board with goal board and using sum of Manhattan distances to decide the next move. This heuristic is not admissible as it is overestimating the value. Using this heuristic we are able to solve all boards, but the moves to solve the board would not be optimal.

Heuristics we tried but did not work:
1. Finding how many numbers are not present in their rows and columns. 
2. Using only the distance of numbers from the rows or columns instead of using both of them.
3. Adding extra high costs for two numbers which are reversed like  "2,1" or "4,3" . It was admissible as the number of moves required to solve a conflict is high, but it was overestimating the value.
  
