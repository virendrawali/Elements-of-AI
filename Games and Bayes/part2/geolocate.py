#!/usr/bin/env python3
from tweet_dataset import read_dataset
from naive_bayes import MultinomialNaiveBayes
import sys

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: ./geolocate.py [training_file] [testing_file] [output_file]")
        exit()

    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    output_file = sys.argv[3]

    _, d = read_dataset(training_file)

    model = MultinomialNaiveBayes(alpha=1e-3)
    model.fit(d.X, d.y)

    texts, d = read_dataset(testing_file, vocabulary=d.vocabulary)
    with open(output_file, "w") as f:
        for tweet, x in zip(texts, d.X):
            predicted_city = d.classes[model.predict(x)]
            print ("{} {}".format(predicted_city, tweet.strip()), file=f)
             
    # prediction = [model.predict(x) for x in d.X]
    # correct = sum(p == y for p, y in zip(prediction, d.y))
    # accuracy = correct * 100/ len(prediction)
    # print ("Accuracy = {0:0.3f}%".format(accuracy))

    for c in model.classes:
        print (d.classes[c])
        print ([d.vocabulary[word_id] for word_id, count in model.distribution_parameters[c].most_common(5)])
