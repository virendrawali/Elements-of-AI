#!/usr/bin/env python3

"""
analysis_of_data.py: computes some statistics on the provided data files

Requirements:
    Python 3

Usage:
    $ python3 analysis_of_data.py
"""

from collections import namedtuple, defaultdict
import csv
import re
import math





print ("city-gps.txt")
print ("="*len("city-gps.txt"))

GeoCoordinates = namedtuple("GeoCoordinates", ["name", "lat", "long"])

with open("city-gps.txt", "r") as handle:
    gps = { name: GeoCoordinates(name, float(latitude), float(longitude)) 
                for name, latitude, longitude in csv.reader(handle, delimiter=" ")}
print ("%s cities." % len(gps))
latitudes = [city.lat for name, city in gps.items()]

print ("latitude: min=%.2f, max=%.2f, mean=%.2f" % (min(latitudes), max(latitudes), sum(latitudes)/len(latitudes)))


longitudes = [city.long for name, city in gps.items()]

print ("longitude: min=%.2f, max=%.2f, mean=%.2f" % (min(longitudes), max(longitudes), sum(longitudes)/len(longitudes)))

print ("\n")

print ("road-segments.txt")
print ("="*len("road-segments.txt"))

RoadSegmentRow = namedtuple("RoadSegment", ["name", "first_city", "second_city", "length", "speed_limit"])
RoadSegment = namedtuple("RoadSegment", ["name", "length", "speed_limit"])


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
                    for first_city, second_city, length, speed_limit, name in rows] 
    

print ("%s road segments." % len(rows))
lengths = [segment.length for segment in rows]
print ("length: min=%.2f, max=%.2f, mean=%.2f" % (min(lengths), max(lengths), sum(lengths)/len(lengths)))

zero_length_segments = [segment for segment in rows if 0 == segment.length]
print ("%s segments with lengths as zero" % len(zero_length_segments))

speed_limits = [segment.speed_limit for segment in rows]
print ("speed_limit: min=%.2f, max=%.2f, mean=%.2f" % (min(speed_limits), max(speed_limits), sum(speed_limits)/len(speed_limits)))


zero_speed_limit_segments = [segment for segment in rows if 0 == segment.speed_limit]
print ("%s segments with speed_limit as zero" % len(zero_speed_limit_segments))

city_names = set(gps.keys())
non_city_segments = [segment for segment in rows if segment.first_city not in city_names or segment.second_city not in city_names]
print ("%s non city segments" % len(non_city_segments))
non_cities = set([segment.first_city for segment in non_city_segments if segment.first_city not in city_names] \
               + [segment.second_city for segment in non_city_segments if segment.second_city not in city_names])

print ("%s cities/junctions without gps coordinate data." % len(non_cities))
