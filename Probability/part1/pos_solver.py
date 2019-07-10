#!/usr/bin/env python3
###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Shivam Rastogi: srastog
# Saurabh Mathur: samathur
# Virendra Wali: vwali
#
# (Based on skeleton code by D. Crandall)
#
####
'''
Key highlights:
Missing words:
	For words missing in the vocabulary of traning data, we add 1e-3
	in the numerator and normalize by a suitable denominator while calculating
	emission probabilities
Missing transitions:
	For all other state, and transition probabilities we add
	1 to the numerator and normalize by a suitable denominator
No. of iterations in MCMC:
	Empirically, we have selected the values as following:
	burnout_time = (length of sentence)*10
	count_iter = (length of sentence)*50
	These values are dependent upon the length of the sentence
Accuracy:
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
         1. Simple:       93.96%               47.50%
            2. HMM:       94.34%               50.65%
        3. Complex:       94.66%               52.65%



State Probability: This is probability of each POS tag in the training data.
State Probability(Si) = (Number of Si tag)/(Total no. of tags in training file)
P(Si) = Count(Si)/sum(Count(Si . . . Sn))
E.g.Calculated using a counter:-
count is Counter({'noun': 236789, 'verb': 146248, 'adp': 123942, '.': 114567})


Initial Probability: Probability of tag being first POS tag in the sentence.
Initial Probability(S1) = (Number of Si being first word of sentence + alpha)/
                          (Total number of POS tags being first word + alpha*Total number of states)


Transition Probability: Probability of transition from one state to another.
So we are calculating transition from one POS tag to another POS tag.

	Transition for viterbi model:
	P(Si+1|Si) = (Count of states (Si+1, Si) + alpha)/(Total number of states (Si) in the training data + alpha*Total number of states)
	Ex:- If Si+1 = Noun, Si = Verb, then we count all occurences of Verb, Noun in the data,
	preserving the order. For denominator, we coung total occurences of Verb in the data

	Transition for MCMC model: Calculated similarly as the viterbi model but
		1 more states is considered.
	P(Si+2|Si+1, Si) = (Count of states (Si+2, Si+1, Si) + alpha)/(Total number of states (Si+1, Si) in the data + alpha*Total number of states)


Emission Probability : It is the probability of observed variable given hidden
state i.e. probability of word given POS tag. Words are the observed variables
and POS tags are hidden variables.
P(Wi|Si) = (Total number of times word wi appear when tag is Si + alpha)/(Total number of times tag Si appears + alpha*Total number of words in vocabulary)


Missing words:
For words in the vocabulary of traning data, we add alpha=1e-3
in the numerator and normalize by a suitable denominator while calculating
emission probabilities. 
Effectively, this assigns a probability = alpha / (Total number of states (Si) in the training data + alpha*Total number of states)


Missing transitions: For all state transition probabilities we add
1 to the numerator and normalize by a suitable denominator
Effectively, this assigns a probability = alpha/ (Total number of states (Si) in the training data + alpha*Total number of states)

This normalization is a consequence of using the bayesian estimate for 
a Multinomial Likelihood and a Dirichlet Prior.


Simplified Model:
	States in simplified model are independent of each other,
	so model is considering POS tag probability and emission probability of word.
	In this model, we are estimating most probable POS tag for each word.


HMM Model:
	For this model, we are using Viterbi algorithm to calculate most probable state
	sequence for given sentence. We are using initial probability,
	transition probability and emission probability.
	Initially we are calculating initial probability for first word of the sentence.
	Vi(Si) = Initial probability of state Si P(Si) * emission probability of word given state P(W1|Si)
	Then we are calculating the probability of each word in the sentence of having particular POS tag.
	Vi = max(P(Si|Si-1) * Vi-1(Si-1))  * P(Wi|Si)

	When we reach at the end of sentence, decide most probable POS tag for last word
	and backtrack to decide which POS tag is most suitable to other words in the sentence.


MCMC Model:
	For calculating the sampling probailities of each state,
	we Consider total 5 terms when sampling any node>2:
	2 nodes before the current nodes + 1 Current node + 2 nodes after the current node.
	Nodes 1 and 2 are sampled and calculated in a different manner.

	No. of iterations: Empirically, we have selected the values as following:
	burnout_time = (length of sentence)*10
	count_iter = (length of sentence)*50
	These values are dependent upon the length of the sentence


Posterior Probability table:
Each row of table gives model and POS structure predicted by it. For example,
the predicted POS sequence for simple model in Row 1.Simple is det  noun adp  noun adv  adp   num  noun  adp  adv.
So to calculate other values of table i.e. Row 1.Simple and Column HMM, we need to find the posterior probability of
POS sequence predicted by simple model in the HMM model. For example, posterior sequence of sentence for simple is
det  noun adp  noun       adv  adp   num  noun  adp, so we need to find this POS sequence in HMM. The prediction made
by model itself will be most probable than other sequence.

		  Simple    HMM      Complex ``   that's all  i'm  sure of   ''   .
Ground truth   	-50.0826  -56.8506  -57.9803 .    prt    prt  prt  adj  adp  .    .
Simple   	-50.0826  -56.8506  -57.9803 .    prt    prt  prt  adj  adp  .    .
HMM	   	-50.0826  -56.8506  -57.9803 .    prt    prt  prt  adj  adp  .    .
Complex 	-52.1689  -57.5632  -57.3713 .    prt    prt  prt  adv  adp  .    .

'''
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
            return self.simplified_diagnostic(sentence, label)
        elif model == "Complex":
            return self.complex_mcmc_diagnostics(sentence, label)
        elif model == "HMM":
            return self.hmm_viterbi_diagnostics(sentence, label)
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

        states = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", ".", "x"]
        self.states = states

        # Calculating all the probabilities using the training data which includes
        # emission, and transition
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
        self.transition_counts = counts
        self.transition_count_totals = totals

        counts = Counter(chain.from_iterable(zip(label_tuple, label_tuple[1:], label_tuple[2:]) for _, label_tuple in data))
        self.bigram_transition_counts = counts

        vocabulary = set(chain.from_iterable([word_tuple for word_tuple, label_tuple in data]))
        self.vocabulary = vocabulary

        counts = Counter(chain.from_iterable([(label, word) for word, label in zip(word_tuple, label_tuple)] for word_tuple, label_tuple in data))
        totals = { state: sum(counts.get((state, word), 0) for word in vocabulary) for state in states }
        self.emission_counts = counts
        self.emission_count_totals = totals


    def _log_state_probability(self, state):
        alpha = 1
        return math.log(self.state_counts.get(state, 0) + alpha) - math.log(self.state_count_total + alpha*len(self.states))

    def _log_initial_state_probability(self, state):
        alpha = 1
        return math.log(self.initial_state_counts.get(state, 0) + alpha) - math.log(self.initial_state_count_total + alpha*len(self.states))

    def _log_transition_probability(self, previous, current):
        count = self.transition_counts.get((previous, current), 0)
        total = self.transition_count_totals.get(previous, 0)
        alpha = 1
        return math.log(count + alpha) - math.log(total + alpha*len(self.states))

    def _log_emission_probability(self, state, observation):
        count = self.emission_counts.get((state, observation), 0)
        total = self.emission_count_totals.get(state, 0)
        alpha = 1e-3
        return math.log(count + alpha) - math.log(total + alpha*len(self.vocabulary))

    def _log_bigram_transition_probability(self, most_previous, previous, current):
        count = self.bigram_transition_counts.get((most_previous, previous, current), 0)
        total = self.transition_counts.get((most_previous, previous), 0)
        return math.log(count + 1) - math.log((total + len(self.states)))

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        def compute_log_posterior_distribution(word):
            return [self._log_state_probability(pos) + self._log_emission_probability(pos, word) for pos in self.states]

        indices = [argmax(compute_log_posterior_distribution(word)) for word in sentence]
        return [self.states[i] for i in indices]


    def simplified_diagnostic(self, sentence, pos_sequence):
        return sum([
            self._log_state_probability(pos_sequence[i])
            + self._log_emission_probability(pos_sequence[i], sentence[i])
            for i in range(len(sentence))
        ])


    def complex_mcmc(self, sentence):

        self.total_samples = 0
        sentence_l = len(sentence)
        pos_tags = []                           #track the current sentence tags

        # Dictionary to save the counts of tags for each word from generated samples
        # It will be used later for finding the tags with maximum counts
        self.posterior_counts = None
        self.posterior_counts = {ind: {state:0 for state in self.states} for ind in range(0, sentence_l)}

        # Assign random states to the sentence
        pos_tags += [self.states[random.randint(0,len(self.states)-2)] for i in range(0, sentence_l)]

        burnout_time = sentence_l*5
        count_iter = sentence_l*20              #num of iterations
        sampling_index = 0                      #save the index being resampled
        # Resample each tag sequentially and save posterior counts
        while(count_iter > 0 ):

            #Store the resampling conditional probabilities for each state
            resampling_probabilities = {}
            denominator = 0

            #Calculate resampled value for each state at position <sampling_index>
            for state in self.states:
                pos_tags[sampling_index]  = state
                numerator = 1
                pr_state_1 = 1
                pr_state_2 = 1

                # When sampling 1st and 2nd node of the graph handle differently
                if(sampling_index == 0):
                    pr_state_3 = 1
                    pr_state_1 = math.exp(self._log_initial_state_probability(pos_tags[0]))*math.exp(self._log_emission_probability(pos_tags[0], sentence[0]))
                    if(sentence_l > 1):
                        pr_state_2 = math.exp(self._log_transition_probability(pos_tags[0], pos_tags[1]))*math.exp(self._log_emission_probability(pos_tags[1], sentence[1]))
                    if(sentence_l > 2):
                        pr_state_3 = math.exp(self._log_transition_probability(pos_tags[1], pos_tags[2]))*math.exp(self._log_emission_probability(pos_tags[2], sentence[2]))
                    numerator = pr_state_1*pr_state_2*pr_state_3

                elif(sampling_index == 1):
                    pr_state_1 = math.exp(self._log_initial_state_probability(pos_tags[0]))*math.exp(self._log_emission_probability(pos_tags[0], sentence[0]))
                    pr_state_2 = math.exp(self._log_transition_probability(pos_tags[0], pos_tags[1]))*math.exp(self._log_emission_probability(pos_tags[1], sentence[1]))
                    numerator *= pr_state_1*pr_state_2

                    #Add pr for node 3,4,5
                    for i in range(sampling_index+1, sampling_index+3):
                        if(i < sentence_l):
                            trans_pr = math.exp(self._log_bigram_transition_probability(pos_tags[i-2],pos_tags[i-1],pos_tags[i]))
                            emis_pr = math.exp(self._log_emission_probability(pos_tags[i], sentence[i]))
                            numerator *= trans_pr * emis_pr

                elif(sampling_index > 1):
                    pr_state_1 = math.exp(self._log_initial_state_probability(pos_tags[0]))*math.exp(self._log_emission_probability(pos_tags[0], sentence[0]))
                    pr_state_2 = math.exp(self._log_transition_probability(pos_tags[0], pos_tags[1]))*math.exp(self._log_emission_probability(pos_tags[1], sentence[1]))
                    numerator *= pr_state_1*pr_state_2

                    # Consider total 5 terms when sampling any node>2:
                    # 2 nodes before the current nodes + 1 Current node + 2
                    # nodes after the current node.
                    for i in range(sampling_index-2, sampling_index+3):
                        if(i < sentence_l):
                            trans_pr = math.exp(self._log_bigram_transition_probability(pos_tags[i-2],pos_tags[i-1],pos_tags[i]))
                            emis_pr = math.exp(self._log_emission_probability(pos_tags[i], sentence[i]))
                            numerator *= trans_pr * emis_pr

                #Add all the numerator values to calculate normalizing factor (denominator)
                denominator += numerator
                resampling_probabilities[state] = numerator

            # Calculate conditional probability for each state at sampling index
            # Normalizing the sampling probabilities
            for key in resampling_probabilities:
                resampling_probabilities[key] = resampling_probabilities[key]/denominator

            # Create list from dictionary for random.choice
            resampling_probabilities_list = []
            for state in self.states:
                resampling_probabilities_list.append(resampling_probabilities[state])

            # Draw a random sample and update the tags of sentence
            draw = n_random.choice(self.states, 1, p=resampling_probabilities_list)
            pos_tags[sampling_index] = draw[0]
            sampling_index = (sampling_index+1)%sentence_l          #Used to decide the location we are sampling
            count_iter -= 1
            burnout_time -= 1

            # Start counting only after the burnout time
            if(burnout_time < 0 ):
                for key in self.posterior_counts:
                    self.posterior_counts[key][pos_tags[key]] += 1
                self.total_samples += 1

        # Assign tags with highest count for each word and return the results
        for i in range(0, len(pos_tags)):
            pos_tags[i] = max(self.posterior_counts[i].keys(), key = lambda x:self.posterior_counts[i][x])

        return pos_tags


    def complex_mcmc_diagnostics(self, sentence, pos_sequence):

        pr_state_1 = 0
        pr_state_2 = 0
        numerator  = 0
        sentence_l = len(sentence)
        if( sentence_l > 0):
            pr_state_1 = self._log_initial_state_probability(pos_sequence[0]) + self._log_emission_probability(pos_sequence[0], sentence[0])
        if(sentence_l > 1):
            pr_state_2 = self._log_transition_probability(pos_sequence[0], pos_sequence[1]) + self._log_emission_probability(pos_sequence[1], sentence[1])

        numerator += pr_state_1 + pr_state_2

        for i in range(2, len(sentence)):
            trans_pr = self._log_bigram_transition_probability(pos_sequence[i-2],pos_sequence[i-1],pos_sequence[i])
            emis_pr = self._log_emission_probability(pos_sequence[i], sentence[i])
            numerator += trans_pr + emis_pr

        return numerator


    def hmm_viterbi(self, sentence):
        states = self.states
        viterbi = [[0 for _ in states] for _ in sentence]
        backtrack = [[0 for _ in states] for _ in sentence]

        # Initialize 0th row with initial_state_probability(state) * emission_probability(word|state)
        viterbi[0] = [self._log_initial_state_probability(state) + self._log_emission_probability(state, sentence[0])
                        for state in states]
        # Steps 2, 3 .. for forward pass
        for i, word in enumerate(sentence[1:], start=1):
            for j, state in enumerate(states):
                options = [viterbi[i-1][p] + self._log_transition_probability(previous, state) for p, previous in enumerate(states)]
                backtrack[i][j] = argmax(options)
                viterbi[i][j] = max(options) + self._log_emission_probability(state, word)

        # Backtracking
        current = argmax(viterbi[-1])
        result = [self.states[current]]
        for row in backtrack[:0:-1]: # all rows except 0th row, in reverse order
            current = row[current]
            result.append(self.states[current])
        return result[::-1]


    def hmm_viterbi_diagnostics(self, sentence, pos_sequence):
        viterbi_given_pos_probabilities = [0] * len(sentence)
        viterbi_given_pos_probabilities[0] = self._log_initial_state_probability(pos_sequence[0])\
                     + self._log_emission_probability(pos_sequence[0], sentence[0])
        for i in range(1, len(sentence)):
            viterbi_given_pos_probabilities[i] = self._log_transition_probability(pos_sequence[i-1], pos_sequence[i])\
                         + self._log_emission_probability(pos_sequence[i], sentence[i])
        return sum(viterbi_given_pos_probabilities)


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
