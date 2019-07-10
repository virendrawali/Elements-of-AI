"""
tweet_dataset.py - contains functions and class definitions needed to 
access and pre-process the given dataset of tweets and locations
"""
from itertools import chain
from collections import namedtuple
import re

city_names = ["Chicago,_IL", "Los_Angeles,_CA", "Toronto,_Ontario", "Orlando,_FL", "Atlanta,_GA", "Boston,_MA",
              "Manhattan,_NY", "Houston,_TX", "Philadelphia,_PA", "San_Francisco,_CA", "Washington,_DC", "San_Diego,_CA"]

def read(file_path):
    """
        Parameters:
            file_path: str, full path to datafile

        Yields:
            a str representing a row containing a tweet's text and its corresponding location
    """
    with open(file_path, "r", encoding="latin-1") as infile:
        tweet = ""
        for line in infile:
            if any(line.startswith(name) for name in city_names):
                if tweet:
                    yield tweet
                tweet = line
            else:
                tweet = tweet + line


def split(row):
    s = row.find(" ")
    location, text = row[:s], row[s+1:]
    return text, location


def tokenize(text):
    return tuple(text.split())

url = re.compile(r"http\S+")
not_allowed = re.compile(r"[^a-z0-9 #'-]")
multiple_space = re.compile(r"\s+")
def clean(text):
    text = text.lower()
    text = url.sub(" ", text)
    text = not_allowed.sub(" ", text)
    text = multiple_space.sub(" ", text)
    return text


def sorted_unique(ls):
    ls = set(ls)
    ls = list(ls)
    ls = list(sorted(ls))
    return ls

TweetDataset = namedtuple("TweetDataset", ["X", "y", "vocabulary", "classes"])

def read_dataset(file_path, vocabulary=None):
    """
        reads and pre-processes the dataset
        
        Parameters:
            file_path: str
            vocabulary: (optional) list of str
        
        Returns:
            a list of raw tweet texts and an instance of TweetDataset

    """
    texts, locations = zip(*(split(row) for row in read(file_path)))
    
    tokens = tuple(tokenize(clean(text)) for text in texts)
    unique_words = sorted_unique(chain.from_iterable(tokens))
    if vocabulary:
        extra_words = set(unique_words) - set(vocabulary)
        vocabulary += sorted_unique(extra_words)
    else:
        vocabulary = unique_words
    
    vocabulary_index = { w: i for i, w in enumerate(vocabulary) }
    X = [[vocabulary_index.get(w) for w in row] for row in tokens]

    classes = sorted_unique(locations)
    y = list(map(classes.index, locations))

    return texts, TweetDataset(
        X=X,
        y=y,
        vocabulary=vocabulary,
        classes=classes
    )