import autograd.numpy as np
import theano
import theano.tensor as T

class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def to_gpu(x):
    return theano.shared(floatX(x))  # TODO this may not be right


def make_batches(N_total, batch_size):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + batch_size))
        start += batch_size
    return batches
