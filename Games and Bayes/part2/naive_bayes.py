from collections import Counter
from itertools import chain
from math import log
from operator import itemgetter
from abc import abstractmethod, ABC


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


class NaiveBayes(ABC):
    def __init__(self, log_priors=None):
        self.log_priors = log_priors
        self.classes = None
        self.distribution_parameters = None

    @abstractmethod
    def _estimate_distribution_parameters(self, partitions):
        pass

    def fit(self, X, y):
        """
            Parameters:
                X: a list of lists, such that each list contains a list of token ids
                y: a list of class ids
        """

        if self.log_priors is None:
            total = len(y)
            class_counts = Counter(y)
            self.log_priors = [log(count) - log(total)
                               for index, count in sorted(class_counts.items())]

        self.classes = list(sorted(set(y)))
        partitions = [[x for x, y_i in zip(X, y) if y_i == c]
                      for c in self.classes]

        self.distribution_parameters = [self._estimate_distribution_parameters(partition)
                                        for partition in partitions]

    @abstractmethod
    def _compute_log_posterior(self, x, y):
        pass

    def predict(self, x):
        """
            Parameters
                x: list of ints

            Returns:
                int, a class id
        """
        log_posterirors = [self._compute_log_posterior(
            x, c) for c in self.classes]
        return argmax(log_posterirors)


class MultinomialNaiveBayes(NaiveBayes):
    def __init__(self, alpha=1, log_priors=None):
        """
            An implementation of the Naive bayes algorithm
            that models the data as a multinomial distribution

            Members:
                alpha: the pseudo-count used for smoothing of the probability values
                log_priors: a dict of class_id (int) -> prior_probability (float)
                distribution_parameters: a list of Counters, 
                    each counter storing the count of data items for a particular class
                classes: a list of ints, representing class ids
        """
        super().__init__(log_priors)
        self.alpha = alpha

    def _estimate_distribution_parameters(self, partition):
        return Counter(chain.from_iterable(partition))

    def _compute_log_posterior(self, x, y):

        counts = self.distribution_parameters[y]
        total, vocabulary_size = sum(counts.values()), len(counts)
        log_likelihood = sum(log(counts.get(x_i, 0) + self.alpha) - log(total + self.alpha*vocabulary_size)
                             for x_i in x)

        return self.log_priors[y] + log_likelihood


#
# Future Work
# ===========
#
# class BernoulliNaiveBayes(NaiveBayes):
#     """
#         An implementation of the Naive bayes algorithm
#         that models the data as a bernoulli distribution
# 
#         Members:
#             alpha: the pseudo-count used for smoothing of the probability values
#             log_priors: a dict of class_id (int) -> prior_probability (float)
#             distribution_parameters: a list of Counters, 
#                 each counter storing the count of data items for a particular class
#             classes: a list of ints, representing class ids
#     """
# 
#     def __init__(self, log_priors=None):
#         super().__init__(log_priors)
# 
#     def _estimate_distribution_parameters(self, partition):
#         vocabulary = set(chain.from_iterable(partition))
#         return { word: log(sum(word in x_i for x_i in partition)) - log(len(partition))
#              for word in vocabulary }
#             
# 