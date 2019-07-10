###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
from numpy import random as n_random
import math
from itertools import chain
from collections import Counter
from operator import itemgetter
import pprint
def argmax(ls):
    """
        returns the index of the maximum value in ls

        Parameters:
            ls: list

        Returns:
            int, the index of maximum value
    """
    ls_with_index = zip(ls, range(len(ls)))
    max_value, index = max(ls_with_index, key=itemgetter(0))
    return index


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        """
            Parameters:
                data(list): a list of (word_tuple, label_tuple)
                each element in label_tuple can be one of
                    1) adj (adjective)
                    2) adv (adverb)
                    3) adp (adposition)
                    4) conj (conjunction),
                    5) det (determiner)
                    6) noun
                    7) num (number)
                    8) pron (pronoun)
                    9) prt (particle)
                    10) verb
                    11) x (foreign word)
                    12) . (punctuation mark)
        """

        #data is a list of tuples, where each tuple has two tuples where contains
        #sentence and another labels
        states = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", "x", "."]
        self.states = states


        counts = Counter(chain.from_iterable(label_tuple for _, label_tuple in data))
        total = sum(counts.values())
        self.state_counts = counts
        self.state_count_total = total


        counts = Counter([label_tuple[0] for _, label_tuple in data])
        total = sum(counts.values())
        self.initial_state_counts = counts
        self.initial_state_count_total = total


        counts = Counter(chain.from_iterable(zip(label_tuple, label_tuple[1:]) for _, label_tuple in data))
        totals = { previous: sum(counts.get((previous, current), 0) for current in states) for previous in states }
        self.transisiton_counts = counts
        self.transition_count_totals = totals

        counts = Counter(chain.from_iterable(zip(label_tuple, label_tuple[1:], label_tuple[2:]) for _, label_tuple in data))
        self.bigram_transisiton_counts = counts

        vocabulary = set(chain.from_iterable([word_tuple for word_tuple, label_tuple in data]))
        self.vocabulary = vocabulary

        counts = Counter(chain.from_iterable([(label, word) for word, label in zip(word_tuple, label_tuple)] for word_tuple, label_tuple in data))
        totals = { state: sum(counts.get((state, word), 0) for word in vocabulary) for state in states }
        self.emission_counts = counts
        self.emission_count_totals = totals


    def _log_state_probability(self, state):
        return math.log(self.state_counts.get(state, 0)) - math.log(self.state_count_total)

    def _log_initial_state_probability(self, state):
        return math.log(self.initial_state_counts.get(state, 0) + 1) - math.log(self.initial_state_count_total + len(self.states))

    def _log_transition_probability(self, previous, current):
        count = self.transisiton_counts.get((previous, current), 0)
        total = self.transition_count_totals.get(previous, 0)
        return math.log(count + 1) - math.log(total + len(self.states))

    def _log_emission_probability(self, state, observation):
        count = self.emission_counts.get((state, observation), 0)
        total = self.emission_count_totals.get(state, 0)
        return math.log(count + 1) - math.log(total + len(self.states))

    def _log_bigram_transition_probability(self, most_previous, previous, current):
        count = self.bigram_transisiton_counts.get((most_previous, previous, current), 0)
        total = self.transisiton_counts.get((most_previous, previous), 0)
        return math.log(count + 1) - math.log((total + len(self.states)))

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        def compute_log_posterior_distribution(word):
            return [self._log_state_probability(pos) + self._log_emission_probability(pos, word) for pos in self.states]

        indices = [argmax(compute_log_posterior_distribution(word)) for word in sentence]
        return [self.states[i] for i in indices]


    def complex_mcmc(self, sentence):
        #print("State pr of x: ",self._log_state_probability('x'))

        sentence_l = len(sentence)
        pos_tags = []                           #save the current tag
        posterior_counts = {ind: {state:0 for state in self.states} for ind in range(0, sentence_l)}
        #Assign random states to the sentence
        pos_tags += [self.states[random.randint(0,len(self.states)-2)] for i in range(0, sentence_l)]

        burnout_time = sentence_l*10
        count_iter = sentence_l*50                        #num of iterations
        sampling_index = 0                      #save the index being resampled
        print(pos_tags)

        #Resample each tag sequentially and save posterior counts
        while(count_iter > 0 ):
            #print("Iteration, sampling_index: ",count_iter, sampling_index)

            curr_state = sentence[sampling_index]
            resampling_probabilities = {}
            denominator = 0

            #print(pos_tags)
            #Calculate resampled value for each state at position <sampling_index>
            for state in self.states[:-1]:
                pos_tags[sampling_index]  = state
                numerator = 1

                pr_state_1 = math.exp(self._log_initial_state_probability(sentence[0]))*math.exp(self._log_emission_probability(pos_tags[0], sentence[0]))
                if(sentence_l > 1):
                    pr_state_2 = math.exp(self._log_transition_probability(pos_tags[0], pos_tags[1]))*math.exp(self._log_emission_probability(pos_tags[1], sentence[1]))
                    numerator *= pr_state_1*pr_state_2
                else:
                    numerator = pr_state_1
                for i in range(2, len(sentence)):
                    #if(i != sampling_index):
                    trans_pr = math.exp(self._log_bigram_transition_probability(pos_tags[i-2],pos_tags[i-1],pos_tags[i]))
                    emis_pr = math.exp(self._log_emission_probability(pos_tags[i], sentence[i]))
                    numerator *= trans_pr * emis_pr

                denominator += numerator
                resampling_probabilities[state] = numerator

            for key in resampling_probabilities:
                resampling_probabilities[key] = resampling_probabilities[key]/denominator
            #pprint.pprint(resampling_probabilities)

            resampling_probabilities_list = []
            for state in self.states[:-1]:
                resampling_probabilities_list.append(resampling_probabilities[state])

            #Draw a random sample and update the tags of sentence
            draw = n_random.choice(self.states[:-1], 1, p=resampling_probabilities_list)
            #print("Value drawn:",  draw)
            pos_tags[sampling_index] = draw[0]

            #print(pos_tags)
            sampling_index = (sampling_index+1)%sentence_l          #Used to decide the location we are sampling
            count_iter -= 1
            burnout_time -= 1

            if(burnout_time < 0 ):
                for key in posterior_counts:
                    posterior_counts[key][pos_tags[key]] += 1

        for i in range(0, len(pos_tags)):
            pos_tags[i] = max(posterior_counts[i].keys(), key = lambda x:posterior_counts[i][x])
        pprint.pprint(posterior_counts)
        print(pos_tags)
        return pos_tags


    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
