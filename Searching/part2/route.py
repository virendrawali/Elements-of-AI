#!/usr/bin/env python3

"""
route.py: finds a route between two cities.

Requirements:
    Python 3

Usage: 
    $ python3 route.py [-h] start_city end_city routing_algorithm cost_function

Assumptions:
    city-gps.txt and road-segments.txt data files reside in the same directory.
    The data in the above two files is not modified. The turns_heuristic and time_heuristic
    use maximum speed limit and maximum path length values derived from the data and the values are hard-coded.
    If the data is changed, those values will have to be changed by finding the values using analysis_of_data.py

Handling Incorrect/Missing data:
    Road segments with length = 0 or speed limit = 0 are ignored and not used for routing.
    
    In the heuristic approximation for cities/junctions without GPS Coordinates,
    the minimum heuristic value from each of their neighbors is used.

"""

import math
from math import sin, cos, radians, sqrt, atan2
import csv
import operator
from queue import PriorityQueue
from collections import namedtuple, defaultdict, deque

GeoCoordinates = namedtuple("GeoCoordinates", ["name", "lat", "long"])
RoadSegmentRow = namedtuple("RoadSegment", ["name", "first_city", "second_city", "length", "speed_limit"])
RoadSegment = namedtuple("RoadSegment", ["name", "length", "speed_limit"])

# for some state s_i,
# g = g(s_i) (path cost)
# f = f(s_i) = g(s_i) + h(s_i)
# path_so_far is a list containing all states
# traversed in current path including current state s_i 
InformedState = namedtuple("InformedState", ["f", "g", "path_so_far"])
UninformedState = namedtuple("UninformedState", ["g", "path_so_far"])

class PriorityQueueFringe(object):
    def __init__(self):
        self._q = PriorityQueue()
    def insert(self, item):
        self._q.put(item)
    def remove(self):
        item = self._q.get()
        return item
    def empty(self):
        return self._q.empty()


class QueueFringe(object):
    def __init__(self):
        self._q = deque()
    def insert(self, item):
        self._q.append(item)
    def remove(self):
        return self._q.popleft()
    def empty(self):
        return not (len(self._q) > 0)

class StackFringe(object):
    def __init__(self):
        self._q = deque()
    def insert(self, item):
        self._q.append(item)
    def remove(self):
        return self._q.pop()
    def empty(self):
        return not (len(self._q) > 0)

def join_segments(s1, s2):
    name = s1.name + " + " + s2.name
    length = s1.length + s2.length
    total_time = s1.length / s1.speed_limit + s2.length / s2.speed_limit
    speed_limit = length / total_time 
    return RoadSegment(name, length, speed_limit)

def approximate_heuristic(heuristic, node, goal):
    # If coordinates available for node, compute heuristic directly
    if node in gps:
        return heuristic(node, goal)

    # otherwise find neighbors which are cities and take minimum heuristic value from each of them
    # as heuristic for current node
    neighbors = connected_nodes[node]
    while not any(neighbor in gps for neighbor, segment in neighbors):
        neighbors = list(set([(n, join_segments(s, s1))  for n1, s1 in neighbors 
                            for n, s in connected_nodes[n1] if n != node] + neighbors))
        
    return min(heuristic(neighbor, goal) for neighbor, segment in neighbors if neighbor in gps)
    

def successors_a_star(state, cost, heuristic, goal):
    f, g, path_so_far = state
    current_node = path_so_far[-1]
    return [InformedState(g + cost(segment)+approximate_heuristic(heuristic, node, goal), g+cost(segment), path_so_far + [node])
        for node, segment in connected_nodes[current_node]]

def successors_uninformed(state, cost):
    g, path_so_far = state
    current_node = path_so_far[-1]    
    return [UninformedState(g + cost(segment), path_so_far + [node])
        for node, segment in connected_nodes[current_node]]

def path_length_cost(segment):
    return segment.length

def turns_cost(segment):
    return 1

def time_cost(segment):
    return segment.length / segment.speed_limit

def euclidean_distance(a, b):
    return math.sqrt((a.lat - b.lat) ** 2 + (a.long - b.long) ** 2)


def path_length_heuristic(node, goal):
    a = gps[node]
    b = gps[goal]
    
    return euclidean_distance(a, b)

def turns_heuristic(node, goal):
    a = gps[node]
    b = gps[goal]
    
    max_road_segment_length = 923 # from data
    # shortest_distance / max_road_segment_length = least number of turns needed
    return euclidean_distance(a, b) / max_road_segment_length

def time_heuristic(node, goal):
    a = gps[node]
    b = gps[goal]

    max_speed_limit = 65 # from data
    # shortest_distance / max_speed_limit = least time needed
    return math.sqrt((a.lat - b.lat)**2 + (a.long - b.long)**2) / max_speed_limit
    
def read_data(gps_filename, road_segments_filename):
    with open(gps_filename, "r") as handle:
        gps = { name: GeoCoordinates(name, float(latitude), float(longitude)) 
                for name, latitude, longitude in csv.reader(handle, delimiter=" ")}
    with open(road_segments_filename, "r") as handle:
        rows = [
            row.strip().split()
            for row in handle
        ]
        rows = [
            row
            for row in rows
            if len(row) == 5
        ] # ignore any rows with missing values
        rows = [RoadSegmentRow(name, first_city, second_city, int(length), int(speed_limit)) 
                        for first_city, second_city, length, speed_limit, name in rows
                        if speed_limit != "0" and length !="0"
                        ] # ignore any rows with incorrect values
        
        connected_nodes = defaultdict(list)
        for (name, first, second, length, speed_limit) in rows:
            connected_nodes[first].append((second, RoadSegment(name, length, speed_limit)))
            connected_nodes[second].append((first, RoadSegment(name, length, speed_limit)))

        connected_nodes_map = defaultdict(dict)
        for (name, first, second, length, speed_limit) in rows:
            connected_nodes_map[first][second] = RoadSegment(name, length, speed_limit)
            connected_nodes_map[second][first] =  RoadSegment(name, length, speed_limit)
    return gps, connected_nodes, connected_nodes_map


def solve_astar(fringe, successors, cost, heuristic, initial_state, goal_state):
    fringe.insert(InformedState(0, 0, [initial_state]))
    closed = set()
    open = {}
    while not fringe.empty():
        state = fringe.remove()
        current_node = state.path_so_far[-1]
        closed.add(current_node)
        if current_node == goal_state:
            return state.g, state.path_so_far
        for s in successors_a_star(state, cost, heuristic, goal_state):
            successor_node = s.path_so_far[-1]
            if successor_node in closed:
                continue
            if successor_node in open:
                if s.g > open[successor_node]:
                    continue 
            open[successor_node] = s.g
            fringe.insert(s)
    return -1, None

def solve_uninformed(fringe, successors, cost, initial_state, goal_state, maximum_depth=None):
    fringe.insert(UninformedState(0, [initial_state]))
    open = {}
    closed = set()
    while not fringe.empty():
        state = fringe.remove()
        current_node = state.path_so_far[-1]

        if current_node == goal_state:
            return state.g, state.path_so_far
        closed.add(current_node)

        depth = len(state.path_so_far)
        if maximum_depth is not None and depth >= maximum_depth:
            continue

        for s in successors(state, cost):
                
            successor_node = s.path_so_far[-1]

            if successor_node in open:
                 if s.g > open[successor_node]:
                    continue 
            open[successor_node] = s.g

            fringe.insert(s)
    
    
    return -1, None


# Globally accessible variables
gps, connected_nodes, connected_nodes_map = read_data("city-gps.txt", "road-segments.txt")


if __name__ == "__main__":
    from argparse import ArgumentParser
    import sys

    metrics = {
        "segments": {
            "cost": turns_cost,
            "heuristic": turns_heuristic
        },
        "distance": {
            "cost": path_length_cost,
            "heuristic": path_length_heuristic
        },
        "time": {
            "cost": time_cost,
            "heuristic": time_heuristic
        }
    }

    fringes = {
        "bfs": QueueFringe,
        "uniform": PriorityQueueFringe,
        "dfs": StackFringe,
        "ids": StackFringe,
        "astar": PriorityQueueFringe
    }

    successors = {
        "bfs": successors_uninformed,
        "uniform": successors_uninformed,
        "dfs": successors_uninformed,
        "ids": successors_uninformed,
        "astar": successors_a_star
    }

    optimal = {
        "bfs": {
            "segments": True,
            "distance": False, 
            "time": False
        },
        "uniform": {
            "segments": True, 
            "distance": True, 
            "time": True
        },
        "ids": {
            "segments": True, 
            "distance": True, 
            "time": True
        },
        "dfs": {
            "segments": False, 
            "distance": False, 
            "time": False
        },
        "astar": {
            "segments": True, 
            "distance": True, 
            "time": True
        }
    }



    supported_algorithms = ["bfs", "uniform", "dfs", "ids", "astar"]
    supported_cost_functions = ["segments", "distance", "time"]

    parser = ArgumentParser()
    parser.add_argument("start_city", metavar="start_city", help="start city")
    parser.add_argument("end_city", metavar="end_city", help="end city")
    parser.add_argument("routing_algorithm", metavar="routing_algorithm", choices=supported_algorithms, help="one of " + ", ".join(supported_algorithms))
    parser.add_argument("cost_function", metavar="cost_function", choices=supported_cost_functions, help="one of " + ", ".join(supported_cost_functions))
    args = parser.parse_args()

    if args.start_city not in gps:
        print ("start_city not found in city-gps.txt")
        sys.exit()
    
    if args.end_city not in gps:
        print ("end_city not found in city-gps.txt")
        sys.exit()

    
    # Maximum depth = Maximum number of road segments
    
    params = dict(fringe=fringes[args.routing_algorithm](),
                  successors=successors[args.routing_algorithm], 
                  cost=metrics[args.cost_function]["cost"], 
                  initial_state=args.start_city, 
                  goal_state=args.end_city)
    
    if args.routing_algorithm == "astar":
        params["heuristic"] = metrics[args.cost_function]["heuristic"]
        g, path = solve_astar(**params)
    elif args.routing_algorithm == "ids":  
        for i in range(12038+1): # Maximum number of road segments
            params["maximum_depth"] = i
            g, path = solve_uninformed(**params)
            if path is not None:
                break
    else:
        g, path = solve_uninformed(**params)

    if path is None:
        print ("No path found.")
    else:
        total_distance = sum(connected_nodes_map[a][b].length
            for a, b in zip(path, path[1:])
        )

        total_time = sum(connected_nodes_map[a][b].length / connected_nodes_map[a][b].speed_limit
            for a, b in zip(path, path[1:])
        )
        print ("%s %s %s %s" % ("yes" if optimal[args.routing_algorithm][args.cost_function] else "no",
                            total_distance,
                            total_time,
                            " ".join(path)))
    

