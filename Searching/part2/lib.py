"""
lib.py: Contains functions that were used to understand the data.
These are no longer used.
approximate_coordinates function has an implementation which we did not have time to optimize
and so it is not used in the current solution
"""

import re
from math import sin, cos, radians, sqrt, atan2
import numpy
import math
from collections import namedtuple, defaultdict
import csv

def euclidean_distance(a, b):
    return math.sqrt((a.lat - b.lat) ** 2 + (a.long - b.long) ** 2)


def extract_node_names(junction_name):
    # For example,
    # junction name = "Jct_US_78_&_US_43/278,_Alabama"
    
    junction_name, state_name = junction_name.split(",_")
    # junction name = "Jct_US_78_&_US_43/278"
    # state_name = "_Alabama"

    junction_name = junction_name.lstrip("Jct_")
    # junction name = "US_78_&_US_43/278"
    
    def decouple_highway_names(name):

        if "/" not in name:
            # For example,
            # name = "Natchez_Trace_Pkwy"
            return [name]

        # For example,
        # name = "US_43/278"

        first, *rest = name.split("/")
        # first = "US_43"
        # rest = ["278"]

        match = re.match(r"([a-zA-Z_-]+)([0-9]+)", first)
        if not match:
            # Error
            print ("Invalid highway name format: \"%s\" " % name)
            return []

        prefix = match.group(1) 
        # prefix = "US_"
        rest = [prefix + each for each in rest]
        return [first] + rest


    highway_names = sum([decouple_highway_names(each) for each in junction_name.split("_&_")], list())

    # Ignoring direction N, S, E, W
    # For example, I-94_N
    return [name.rstrip("_NSEW") for name in highway_names]



def haversine_distance(city1, city2):
    # Reference : https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
    
    R = 3960.0

    lat1 = radians(city1.lat)
    lon1 = radians(city1.long)
    lat2 = radians(city2.lat)
    lon2 = radians(city2.long)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def triangulate(a, a_dist, b, b_dist, c, c_dist):
    # reference: https://gis.stackexchange.com/questions/66/trilateration-using-3-latitude-longitude-points-and-3-distances
    earthR = 3960.0
    LatA = a.lat
    LonA = a.long
    DistA = a_dist
    LatB = b.lat
    LonB = b.long
    DistB = b_dist
    LatC = c.lat
    LonC = c.long
    DistC = c_dist

    #using authalic sphere
    #if using an ellipsoid this step is slightly different
    #Convert geodetic Lat/Long to ECEF xyz
    #   1. Convert Lat/Long to radians
    #   2. Convert Lat/Long(radians) to ECEF
    xA = earthR *(math.cos(math.radians(LatA)) * math.cos(math.radians(LonA)))
    yA = earthR *(math.cos(math.radians(LatA)) * math.sin(math.radians(LonA)))
    zA = earthR *(math.sin(math.radians(LatA)))

    xB = earthR *(math.cos(math.radians(LatB)) * math.cos(math.radians(LonB)))
    yB = earthR *(math.cos(math.radians(LatB)) * math.sin(math.radians(LonB)))
    zB = earthR *(math.sin(math.radians(LatB)))

    xC = earthR *(math.cos(math.radians(LatC)) * math.cos(math.radians(LonC)))
    yC = earthR *(math.cos(math.radians(LatC)) * math.sin(math.radians(LonC)))
    zC = earthR *(math.sin(math.radians(LatC)))

    P1 = numpy.array([xA, yA, zA], dtype=numpy.float64)
    P2 = numpy.array([xB, yB, zB], dtype=numpy.float64)
    P3 = numpy.array([xC, yC, zC], dtype=numpy.float64)

    #from wikipedia
    #transform to get circle 1 at origin
    #transform to get circle 2 on x axis
    ex = (P2 - P1)/(numpy.linalg.norm(P2 - P1))
    i = numpy.dot(ex, P3 - P1)
    ey = (P3 - P1 - i*ex)/(numpy.linalg.norm(P3 - P1 - i*ex))
    ez = numpy.cross(ex,ey)
    d = numpy.linalg.norm(P2 - P1)
    j = numpy.dot(ey, P3 - P1)

    #from wikipedia
    #plug and chug using above values
    x = (pow(DistA,2) - pow(DistB,2) + pow(d,2))/(2*d)
    y = ((pow(DistA,2) - pow(DistC,2) + pow(i,2) + pow(j,2))/(2*j)) - ((i/j)*x)

    # only one case shown here
    z = numpy.sqrt(pow(DistA,2) - pow(x,2) - pow(y,2))

    #triPt is an array with ECEF x,y,z of trilateration point
    triPt = P1 + x*ex + y*ey + z*ez

    #convert back to lat/long from ECEF
    #convert to degrees
    lat = math.degrees(math.asin(triPt[2] / earthR))
    lon = math.degrees(math.atan2(triPt[1],triPt[0]))

    return lat, lon


def approximate_coordinates():
    """
    This was an attempt at imputing the missing GPS coordinate values.
    However, due to inconsistencies and incorrect entries in the original data,
    all imputed values were not correct. As a result, this approach was not used in the final solution.
    
    While we didn't have time to implement it, non linear squares approximation method
    would account for errors in the original data.
    Reference: https://appelsiini.net/2017/trilateration-with-n-points/


    """
    RoadSegmentRow = namedtuple("RoadSegment", ["name", "first_city", "second_city", "length", "speed_limit"])
    RoadSegment = namedtuple("RoadSegment", ["name", "length", "speed_limit"])
    GeoCoordinates = namedtuple("GeoCoordinates", ["name", "lat", "long"])
    with open("city-gps.txt", "r") as handle:
        gps = { name: GeoCoordinates(name, float(latitude), float(longitude)) 
                    for name, latitude, longitude in csv.reader(handle, delimiter=" ")}
        with open("road-segments.txt", "r") as handle:
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
                            if length != "0" and speed_limit != "0"] 
    

    city_names = set(gps.keys())
    non_city_segments = [segment for segment in rows if segment.first_city not in city_names or segment.second_city not in city_names]
    print ("%s non city segments" % len(non_city_segments))
    non_cities = set([segment.first_city for segment in non_city_segments if segment.first_city not in city_names] \
                + [segment.second_city for segment in non_city_segments if segment.second_city not in city_names])

    connected_nodes = defaultdict(list)
    for (name, first, second, length, speed_limit) in rows:
        connected_nodes[first].append((second, RoadSegment(name, length, speed_limit)))
        connected_nodes[second].append((first, RoadSegment(name, length, speed_limit)))
    #
    # Iteratively approximate coordinates of nodes whose coordinates are unknown
    #
    while len(non_cities) > 0:
        # Create an ordered list of nodes still unknown
        
        non_cities_with_neighbour_count = []
        for node in non_cities:
            connected_cities = [n for n, segment in connected_nodes[node] if n in gps]
            non_cities_with_neighbour_count.append((len(connected_cities), node))

        # Iterate over remaining unknown nodes in descending order of count of neighbouring known nodes

        for _, non_city in sorted(non_cities_with_neighbour_count, reverse=True):
            
            # Expand paths till depth 5 and find all connected nodes which are known

            next_nodes = [(n, s.length) for n, s in connected_nodes[non_city]]
            
            max_depth = 5
            for _ in range(max_depth):
                next_nodes = list(set([(n, s.length + l1) for n1, l1 in next_nodes 
                                for n, s in connected_nodes[n1] if n != non_city] + next_nodes))
            
            
            connected_cities_with_distance = [(segment.length, n) 
                for n, segment in connected_nodes[non_city] 
                if n in gps]
            
            # Take top k known nodes with least distance from current unknown node

            k=3
            top_k_cities = [(n, l) for l, n in sorted(connected_cities_with_distance)][:k]
            
            if len(top_k_cities) == 0:
                continue
            
            # Compute the latitude and longitude of current node 
            # by averaging corresponding values of known nodes
            
            if len(top_k_cities) == 3:
                latitude, longitude = triangulate(gps[top_k_cities[0][0]], top_k_cities[0][1],
                                                gps[top_k_cities[1][0]], top_k_cities[1][1],
                                                gps[top_k_cities[2][0]], top_k_cities[2][1])
                
            
            if len(top_k_cities) != 3 or numpy.isnan(latitude) or numpy.isnan(longitude):
                print ("hello")
                top_k_cities = [n for n, l in top_k_cities]
                latitude = sum(gps[city].lat for city in top_k_cities) / len(top_k_cities)
                longitude = sum(gps[city].long for city in top_k_cities) / len(top_k_cities)
            
            

            # Add current node's newly computed location to the gps dict 
            gps[non_city] = GeoCoordinates(non_city, latitude, longitude)
            
            # mark current node as known
            non_cities.remove(non_city)
            

    # write the gps dict to a file
    with open("city-gps-corrected.txt", "w") as handle:
        writer = csv.writer(handle, delimiter=" ")
        for location in gps.values():
            writer.writerow([location.name, location.lat, location.long])





    with open("road-segments-corrected.txt", "w") as handle:
        writer = csv.writer(handle, delimiter=" ")
        for row in rows:
            distance = euclidean_distance(gps[row.first_city], gps[row.second_city])
            if distance > row.length:
                # These observations have errors in them
                # as observed path length cannot be less than euclidean distance 
                continue
            writer.writerow([row.first_city, row.second_city, row.length, row.speed_limit, row.name])
        