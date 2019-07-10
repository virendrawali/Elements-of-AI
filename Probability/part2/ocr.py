#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: 
# Saurabh Mathur (samathur)
# Shivam Rastogi (srastog)
# Virendra Wali (vwali)
# (based on skeleton code by D. Crandall, Oct 2018)
#
"""
The key difference from part1 is in the computation of the emission probabilities.
P(Image|character=c) is computed making the naive bayes assumption that each pixel is 
independent of the other pixels given the character.

So, P(Image|character=c) = \prod_{i}^{width} \prod_{j}^{height} P(pixel(i,j) matches original image's pixel(i,j))

Let pixel_matched be a random variable that is 1 if the pixel(i,j) in current image matches the pixel(i,j) in original image.
pixel_matched follows a Bernoulli distribution with parameter \mu.
Therefore, for an image which is a collection of i.i.d. pixels, the likelihood follows 
the Binomial distribution with parameter \mu and n = height*width of the image.

Concretely, we have likelihood ~ Bernoulli(n, \mu)

We can take a conjugate prior for \mu ie, the beta distribution with pseudocounts \alpha and \beta

So, we get prior, \mu ~ Beta(\alpha, \beta)

Thus, The posterior can be computed as posterior = prior*likelihood*C for some constant C. 
On simplifying, we can write the posterior as posterior ~ Beta(alpha+x, beta+n-x)
On solving for the bayesian solution (predictive distribution), we get 

P(pixel-matched|mu) = \mu = (alpha + x) / (alpha + beta + n)
    where x is the number of matches seen in training image 
    and n the size of training image. 

However, since we have not seen any training data, we fall back on the prior and the new expression is 
P(pixel-matched|mu) = \mu =  (alpha) / (alpha + beta)

Things we tried:
1.  Initially, we used the simplest prior for all the characters. So, the value of \mu for each character was the same (0.7).
    After observing that some characters are much more likely to get corrupted, we set the value of \mu for space and some punctuation marks to a smaller value.
    While this yielded better results, the results could be further improved by
    1. More careful tuning of \mu for each letter
    2. The use of more training images that are corrupted with similar type of noise to compute \mu using the predictive distribution.
    We also ignored the top and the bottom "rows" of pixels since those rows were whitespace and didn't contain any information.
2.  We also tried to use MCMC for prediction but it turns out that the character dependencies are deeper than our model for MCMC. Since our Bayes net for MCMC only takes into consideration previous two parent nodes, the model was not able to perform very well. Viterbi algorithm is superior than MCMC and hence we have used it. 

Model Summary:
Assumption: Each pixel is conditionally independent given the observed character
    Likelihood: pixel-matched ~ Binomial(n, mu)
    Prior: mu ~ Beta(alpha, beta)
    Posterior: mu|data ~ Beta(alpha+x, beta+n-x)
    Bayesian Solution: P(pixel-matched|mu) =  (alpha + x) / (alpha + beta + n)
    where x: number of pixels that matched
          n: total number of pixels
      alpha: pseudocount for matched pixels
       beta: pseudocount for pixels that were not matched
    Here, we have seen no data at all
    So, P(pixel-matched|mu) =  (alpha) / (alpha + beta)
"""

from PIL import Image, ImageDraw, ImageFont
import sys
import math
from itertools import chain
from operator import itemgetter
from collections import Counter
from pprint import pprint 


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


CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(
            x_beg, x_beg+CHARACTER_WIDTH)]) for y in range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}

def load_text_lines(fname):
    with open(fname, "r") as handle:
        return [line.strip() for line in handle if len(line) > 2]
        


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
lines = load_text_lines(train_txt_fname)

# Below is just some sample code to show you how the functions above work.
# You can delete them and put your own code here!

states = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


state_counts = Counter(chain.from_iterable(line for line in lines))
state_count_total = sum(state_counts.values())


initial_state_counts = Counter(chain.from_iterable(line[0] for line in lines))
initial_state_count_total = sum(initial_state_counts.values())


lines = lines + [line.upper() for line in lines]
transition_counts = Counter(chain.from_iterable(zip(line, line[1:]) for line in lines))
transition_count_totals = { previous: sum(transition_counts.get((previous, current), 0) for current in states) for previous in states } 



def log_emission_probability(state, observation):
    # Assumption: Each pixel is conditionally independent given the observed character
    # Likelihood: pixel-matched ~ Binomial(n, mu)
    # Prior: mu ~ Beta(alpha, beta)
    # Posterior: mu|data ~ Beta(alpha+x, beta+n-x)
    # Bayesian Solution: P(pixel-matched|mu) =  (alpha + x) / (alpha + beta + n)
    # where x: number of pixels that matched
    #       n: total number of pixels
    #   alpha: pseudocount for matched pixels
    #    beta: pseudocount for pixels that were not matched
    # Here, we have seen no data at all
    # So, P(pixel-matched|mu) =  (alpha) / (alpha + beta)

    mu = .7  
    # !Improvement: Different values of alpha and beta for each character
    # Since 1 and l are more likely to get corrupted by noise than A or Z

    height, width = len(train_letters[state]), len(train_letters[state][0])
    n = height * width
    
    if state in ' ,.\'"-':
        # The punctuation marks can be confused with noise
        mu = .65
    
    

    matches = sum(new_value == original_value for original_value, new_value in
                  zip(chain.from_iterable(train_letters[state][1:-1] ), chain.from_iterable(observation[1:-1])))
    
    
    return matches * math.log(mu) + (n-matches) * math.log(1-mu)


def log_initial_state_probability(state):
    alpha = 1
    return math.log(initial_state_counts.get(state, 0) + alpha) - math.log(initial_state_count_total + alpha*len(states))

def log_state_probability(state):
    alpha = 1
    return math.log(state_counts.get(state, 0) + alpha) - math.log(state_count_total + alpha*len(states))

def log_transition_probability(previous, current):
    count = transition_counts.get((previous, current), 0)
    total = transition_count_totals.get(previous, 0)
    alpha = 1
    return math.log(count + alpha) - math.log(total + alpha*len(states))



def simplified(images):
    def compute_log_posterior_distribution(image):
        return [log_state_probability(c) + log_emission_probability(c, image) for c in states]

    indices = [argmax(compute_log_posterior_distribution(image)) for image in images]
    return [states[i] for i in indices]

def hmm_viterbi(images):
    # states = states
    viterbi = [[0 for _ in states] for _ in images]
    backtrack = [[0 for _ in states] for _ in images]


    # Initialize 0th row with initial_state_probability(state) * emission_probability(image|state)
    viterbi[0] = [log_initial_state_probability(state) + log_emission_probability(state, images[0])
                    for state in states]
    # Steps 2, 3 .. for forward pass
    for i, image in enumerate(images[1:], start=1):
        for j, state in enumerate(states):
            options = [viterbi[i-1][p] + log_transition_probability(previous, state) for p, previous in enumerate(states)]
            backtrack[i][j] = argmax(options)
            viterbi[i][j] = max(options) + log_emission_probability(state, image)
    
    # Backtracking
    current = argmax(viterbi[-1])
    result = [states[current]]
    for row in backtrack[:0:-1]: # all rows except 0th row, in reverse order
        current = row[current]
        result.append(states[current])
    return result[::-1]

print ("".join(simplified(test_letters)))
print ("".join(hmm_viterbi(test_letters)))
