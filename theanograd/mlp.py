import autograd.numpy as np
from autograd.core import primitive
from autograd import grad

import theano
import theano.tensor as T


###########################
# theano helper functions #
###########################

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def to_gpu(x):
    return theano.shared(floatX(x))  # TODO this may not be right

#########################################
# Define MLP primitive and its gradient #
#########################################

def make_mlp(shapes):
    """ Make a multilayer perceptron function where we get gradients from 
    optimized theano code.  This returns a function handle

        mlp = make_mlp(shapes)

    where mlp(input, params) pushes inputs through a multi-layer perceptron
    with params = [(W1, b1), (W2, b2), ..., (WL, bL)] a list of weights and
    biases.

    Autograd will be able to differentiate functions that use MLP with theano's
    gradients
    """
    def pack(params):
        return np.concatenate([np.concatenate([np.ravel(W), b])
                               for W, b in params])

    def unpack(params):
        offset = 0
        for m, n in shapes:
            yield params[offset:offset+m*n].reshape((m,n)), params[offset+m*n:offset+(m+1)*n]
            offset += (m+1)*n

    def mlp(x, params):
        for W, b in unpack(params):
            x = T.tanh(T.dot(x, W) + b)
        return x

    # define the MLPs jacobian-vector product using theano
    params  = T.dvector('params')
    x       = T.dmatrix('x')
    g       = T.dmatrix('g')
    mlpval  = mlp(x, params)
    gradfun = theano.function([x, params, g], T.Lop(mlpval, params, g))

    # create python executable MLP function, define autograd primitive
    theano_mlpfun = theano.function([x, params], mlpval)
    mlpfun = primitive(lambda x, params: theano_mlpfun(x, pack(params)))
    mlpfun.defgrad(lambda ans, x, params: lambda g: list(unpack(gradfun(x, pack(params), g))), 1)

    return mlpfun


if __name__=="__main__":
    shapes = [(2, 10), (10, 2)]
    mlp    = make_mlp(shapes)

    def fun(x, y, params):
        yhat = mlp(x, params)
        return np.sum((yhat - y)**2)

    def fun_notheano(x, y, params):
        for W, b in params:
            x = np.tanh(np.dot(x, W) + b)
        return np.sum((x-y)**2)

    x      = np.random.randn(100, 2)
    y      = np.random.randn(100, 2)
    params = [(np.random.randn(s[0], s[1]), np.random.randn(s[1])) for s in shapes]

    print fun_notheano(x, y, params)
    print fun(x, y, params)

    print grad(fun_notheano, 2)(x, y, params)
    print grad(fun,2)(x, y, params)

