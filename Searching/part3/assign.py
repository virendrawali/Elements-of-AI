#!/usr/bin/env python3
import pprint
import sys
import csv
import queue
from collections import namedtuple

'''
Our design is focused on giving the best possible cost in a short amount of
time. We start with an admissible heuristic function and make it greedier as the
number of visited states reach a certain threshold. The heuristic keeps on
becoming greedier as the number of states increase which ensures that it will
always provide a result even though it might be sub-optimal

## Initial State
Initial state contains one team with one student in it

## Goal State - Goal state is when all the students are present in a state we
declare it a goal state. We rely on the algorithm to reach to the closest goal,
and we declare the first state that has all the students in it as goal state.

## Successor function - The successor function receives a state from the fringe.
Then it takes a student at a time from the remaining students and creates all
possible permutations of the state with that student. Each time all the
remaining students are permuted to create successors.
For example: If there are 3 students A, B and C. Initial board has [[A]],
then the successor will generate 4 states that would be :
                      [[AB]], [[A][B]], [[AC]], [[A],[C]]
                      In the next iteration if we take out [[AB]] from the fringe
                      then the states would be:
                      [[ABC]], [[AB],C]

## Cost function
# --------------------------------------------------------------------------- #
make_greedy = (No. of students)^(int(num_visited_states/(total_num_stud*40000)))

return (state_data.time_required/make_greedy + heuristic(state_data)/make_greedy)
# --------------------------------------------------------------------------- #
Since each state in our algorithm contains a different number of
students, it was a challenge to decide the appropriate cost function. We tried
different methods of normalizing the cost so that as we reach closer to the goal
state the cost of the state reduces. For example, if we have two states with
the same number of conflicts but a different number of students the algorithm
should pick the one with the higher number of students. It looks simple, but if
we don't normalize the cost function, the program ends up visiting too many
states.
Our cost function uses values of the arguments k,m and n to evaluate each state.
We normalize the value of cost function by dividing it with the number of
students in that state. The challenge we have faced by using this approach is
that the cost function starts giving similar values for multiple states as the
number of students in the state increases. In order to tackle it, we have
designed an algorithm that becomes greedier as the number of iterations increase.
# Switch to greedy algorithm
As the number of states reach a certain threshold, our cost function changes.
We start giving more preference to states that have more students. For example:
If the number of students is 5, then after 5*40000=80000 states our cost
function will increase the denominator value represented by variable
'make_greedy'. Therefore, the algorithm becomes greedier and starts descending
faster towards the goal state.


## Assumptions:
1. Value of k is larger than m and n. If the value of k is smaller
algorithm visits too many states and slowly it will become greedier as the
number of iterations would increase(as per our design).
2. The program would be run for at least 15 minutes if the number of students
is more than 50.
3. There is a trade-off between the time taken to reach the solution and the
cost of the goal state. For optimal costs, the program might have to be run for
many hours and in order to avoid that we slowly make our algorithm greedy.
Therefore, we get sub-optimal solutions in a relatively shorter amount of time
which can be controlled from within the script.


## Heuristic function
It estimates the number of students who won't be able to get into their
'preferred partners' in the goal state as the 'preferred partners' are already
in a team of 3 people.
For example:
In state [[ABC]], if we want to add D to our state, only possible state as per
our successor function is [[ABC][D]]. In this case, if D wants to work with A
it will never be able to do so. Therefore our heuristic will add cost which
 will be equal to parameter 'n' times 1.


## Things we tried:
1. We have tried out different cost functions but since we start with an initial
state of one person, our biggest challenge was to normalize the costs between
two states with a different number of students in them. We have tried many cost
functions like using the ratio of k, m, and n to normalize the costs.
2. We also tried different approaches of creating successor states like sorting
the students in a specific order(students with most restrictions first and then
students with less restrictions). After sorting, we would use the list to pop
one student each time and create the states using that student so that at every
level only student was added. It did not work and random order of the list was
giving us better results. Therefore, we have switched to the current design in
which level we use all the available students to create our successors at
every level.'''



#Calculate time required by the state as per the arguments
def calculate_time_required(state, k_per_team = 1, m_not_prefer_team = 1, n_prefer_team = 1):
    all_students_in_state = get_stud_from_state(state)
    cost = 0
    num_stud_state = 0

    # Each state is a list of list where inner list represents a team.
    # Therefore, counting number of elements in one state will give the total
    # number of teams.
    cost += k_per_team*len(state)
    for team in state:
        for stud in team:
            num_stud_state = num_stud_state + 1
            if(stud_pref_dict[stud]['prefer_team'][0] != '_'):
                for prefer in stud_pref_dict[stud]['prefer_team']:
                    if(prefer not in team):
                        cost += n_prefer_team

            if(stud_pref_dict[stud]['not_prefer_team'][0] != '_'):
                for not_prefer in stud_pref_dict[stud]['not_prefer_team']:
                    if(not_prefer in team):
                        cost += m_not_prefer_team

            if (int(stud_pref_dict[stud]['num_team_mem'][0]) != 0):
                if(len(team) != int(stud_pref_dict[stud]['num_team_mem'][0])):
                    cost += 1

    return (cost, num_stud_state, all_students_in_state)


# Heurisitic functions checks how many students who are requested by other
# students are already in a team of 3 students and wont be available again.
def heuristic(state_data):
    team_with_3_people = []

    for team in state_data.state:
        if len(team ) ==  3:
            team_with_3_people += team
    count = 0

    for student in all_students:
        if student not in state_data.studs_in_state:
            for prefer_stud in stud_pref_dict[student]['prefer_team']:
                if prefer_stud in team_with_3_people:
                    count = count + 1
    return n*count


# Check if all the students are present in the state
def is_goal(state):
    all_students_in_state = []
    count = 0

    for team in state:
        for assigned_student in team:
            all_students_in_state.append(assigned_student)
            count = count+1
    for student in all_students:
        if(student not in all_students_in_state):
            return False
    return True


# Return a list of possible successor states
def successors2(state):
    all_students_in_state = get_stud_from_state(state)
    next_student= ""
    succ_states = list()

    for student in all_students:
        if (student not in all_students_in_state):
            next_student = student
            succ_states += [state + [[next_student]]]
            for i,e in enumerate(state):
                if(len(state[i]) < 3  ):
                    succ_states +=  [state[:i] + [state[i]+ [next_student]] + state[i+1:]]
    return succ_states


# Extract a list of students from the state
def get_stud_from_state (state):
    all_students_in_state = []

    for team in state:
        for assigned_student in team:
            all_students_in_state.append(assigned_student)
    return all_students_in_state


# Calculate g(s) and h(s), i.e adds the cost function value and the heuristic
# cost value of the state
def state_cost(state_data, num_visited_states):
    make_greedy =  \
          state_data.num_stud**(int(num_visited_states/(total_num_stud*40000)))

    return (state_data.time_required/make_greedy \
	 + heuristic(state_data)/make_greedy) \


def print_goal(state_data):

    for team in state_data.state:
        print(' '.join(team))


def solve(initial_board):
    num_visited_states = 0

    time_required, num_stud, stud_in_state  = \
                                calculate_time_required(initial_board, k, m, n)
    initial_state_data = State_details(initial_board, \
                                        stud_in_state, time_required, num_stud )
    fringe.put(  ( state_cost(initial_state_data, num_visited_states), \
                                                        initial_state_data )  )
    while not fringe.empty():
        cost, state_data = fringe.get()

        if(str(state_data.state) in closed):
            continue

        closed.add(str(state_data.state))
        if is_goal(state_data.state):
            return(state_data)

        for succ in successors2( state_data.state ):
            num_visited_states += 1
            time_required, num_stud, stud_in_succ = calculate_time_required(succ, k, m, n)
            succ_data = State_details(succ, stud_in_succ, time_required, num_stud )
            fringe.put((state_cost(succ_data, num_visited_states), succ_data))


# ------------------------------- Main Program ---------------------------------
fringe = queue.PriorityQueue()
State_details = namedtuple('State_details', ['state', 'studs_in_state',\
                                                 'time_required', 'num_stud', ])
input_data_dict = dict()
all_students = []
total_num_stud = 0
closed = set()

#command line arguments
k = int(sys.argv[2])
m = int(sys.argv[3])
n = int(sys.argv[4])


# Code for opening csv file is taken from the following source:
#https://stackoverflow.com/questions/35829360/python-read-csv-file-with-row-and-column-headers-into-dictionary-with-two-keys
with open(sys.argv[1], "r") as infile:
    stud_pref_dict=dict()
    reader = csv.reader(infile, delimiter=" ")
    headers = ["num_team_mem", "prefer_team", "not_prefer_team"]
    for row in reader:
        stud_pref_dict[row[0]] = {key: (value.split(',')) \
                                    for key, value in zip(headers, row[1:])}
        all_students += [row[0]]

total_num_stud = len(all_students)

# Initialize the initial state with one student from the list. Each state is
# represented in a list of list, where the inner list count represents one team
initial_state = [[all_students[0]]]

goal_state_data = solve(initial_state)

print_goal(goal_state_data)
print(goal_state_data.time_required)
