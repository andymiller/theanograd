"""Autograd+Theano convnet - get gradients for a convolutional MLP using
theano, manipulate it at a higher level with autograd.
"""
import autograd.numpy as np
import autograd.scipy.misc as scpm
from autograd.core import primitive
from autograd.util import quick_grad_check
from autograd import grad

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from util import floatX, to_gpu, WeightsParser, make_batches


#############################
# Convnet helper classes    #
#############################
class full_layer(object):
    def __init__(self, size):
        self.size = size

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.w_shp = (input_size, self.size)
        self.b_shp = (self.size, )
        self.num_bias_params = self.size
        self.num_parameters  = self.num_bias_params + self.size * input_size
        return self.num_parameters, (self.size,)

    def forward_pass(self, inputs, params):
        b = params[:self.num_bias_params].reshape(self.b_shp)
        W = params[self.num_bias_params:].reshape(self.w_shp)
        inputs = inputs.reshape((inputs.shape[0], -1))
        x = self.nonlinearity(T.dot(inputs, W) + b)
        return x

class tanh_layer(full_layer):
    def nonlinearity(self, x):
        return T.tanh(x)

class softmax_layer(full_layer):
    def nonlinearity(self, x):
        return x - scpm.logsumexp(x, axis=1, keepdims=True)

class conv_layer(object):

    def __init__(self, filter_shape, num_filters, pool_shape=(2, 2)):
        """ initialize layer with filter shape (eg 5x5), 
        number of filters (= number of output maps for this layer)
        number of input_maps (eg how many color/feature channels the input has)
          i.e. w_shp = (num_filters, num_input, filt_rows, filt_cols)
        """
        self.filter_shape = filter_shape
        self.num_filters  = num_filters
        self.pool_shape   = pool_shape

    def build_weights_dict(self, input_shape):
        # compute weights and bias shape, given input shape
        num_input_maps = input_shape[0]
        self.w_shp = (self.num_filters, num_input_maps) + self.filter_shape
        self.b_shp = (self.num_filters, )
        self.num_filter_weights = np.prod(self.w_shp)
        self.num_params         = self.num_filter_weights + self.b_shp[0]

        # compute output shape
        output_shape = (self.num_filters, ) + \
                       self.conv_output_shape(input_shape[1:], self.filter_shape)
        self.output_shape = output_shape
        return self.num_params, output_shape

    def forward_pass(self, inputs, params):
        # Input dimensions:  [data, color_in, y, x]
        # Params dimensions: [color_in, color_out, y, x]
        # Output dimensions: [data, color_out, y, x]
        W = params[:self.num_filter_weights].reshape(self.w_shp)
        b = params[self.num_filter_weights:]

        # convolve input
        conv_out = T.nnet.conv.conv2d(inputs, W)

        # pool conv out
        pool_out = downsample.max_pool_2d(conv_out, self.pool_shape, ignore_border=True)

        # push output through a nonlinearty and return
        output   = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)

###############################################################
# Make autograd differentiable convolutional MLP function     #
###############################################################

def make_conv_mlp(input_shape, layers):
    """ Make a convolutional multilayer perceptron function where we get 
    gradients from optimized theano code.  This returns a function handle

        mlp = make_mlp(shapes)

    where mlp(input, params) is a function of 

      - inputs = tensor of size
              [num data,
               number of input feature maps (color-in),
               image height,
               image width]

      - params decomposes into a list of parameter tensors, the convolutional 
        tensor size is:

              [number of feature maps at output (color-out), (number of filters?)
               number of feature maps at input (color-in),
               filter height,
               filter width]

    Autograd will be able to differentiate functions that use MLP with theano's
    gradients
    """
    # compute number of params and output shapes for each layer - compile a
    # list of
    offset = 0
    param_slices = []
    cur_shape = input_shape
    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        param_slices.append(slice(offset, offset+N_weights))
        offset += N_weights
    num_params = offset
    out_size   = np.prod(cur_shape)

    def unpack(params):
        for pslice in param_slices:
            yield params[pslice]

    def mlp(inputs, params):
        """applies each layer to the input, given parameter vector.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for layer, layer_params in zip(layer_specs, unpack(params)):
            cur_units = layer.forward_pass(cur_units, layer_params)

        # make sure we're returning a 2-d ... for theano Lop
        return cur_units.reshape((inputs.shape[0], -1))

    # define symbolic variables that theano will manipulate
    inputs  = T.tensor4(name='inputs', dtype='float32')
    params  = T.fvector('params')
    g       = T.fmatrix('g')

    # define the mlp symblic function and executable function
    mlpval        = mlp(inputs, params)
    theano_mlpfun = theano.function([inputs, params], mlpval, allow_input_downcast=True)

    # define the Jacobian-Vector gradient function needed for autograd
    gradfun = theano.function([inputs, params, g], T.Lop(mlpval, params, g), allow_input_downcast=True)

    # create python executable MLP function, define autograd primitive
    mlpfun = primitive(lambda x, params: theano_mlpfun(x, params))
    mlpfun.defgrad(lambda ans, x, params: lambda g: gradfun(x, params, g), 1)
    return mlpfun, num_params, out_size


if __name__ == '__main__':

    if 'train_images' not in locals():
        # Load and process MNIST data (borrowing from Kayak)
        print("Loading training data...")
        import imp, urllib
        add_color_channel = lambda x : x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
        one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
        source, _ = urllib.urlretrieve(
            'https://raw.githubusercontent.com/HIPS/Kayak/master/examples/data.py')
        data = imp.load_source('data', source).mnist()
        train_images, train_labels, test_images, test_labels = data
        train_images = add_color_channel(train_images) / 255.0
        test_images  = add_color_channel(test_images)  / 255.0
        train_labels = one_hot(train_labels, 10)
        test_labels  = one_hot(test_labels, 10)
        N_data       = train_images.shape[0]


    ####################################################
    # define a convolutional multi-layer perceptron    #
    ####################################################
    L2_reg = 1.0
    input_shape = (1, 28, 28)
    layer_specs = [
        conv_layer(filter_shape = (5, 5), num_filters=6, pool_shape=(2,2)),
        conv_layer(filter_shape = (5, 5), num_filters=16, pool_shape=(2,2))
    ]

    # Make neural net functions, define prediction function
    mlp, num_conv_params, out_size = make_conv_mlp(input_shape, layer_specs)

    # make fully connected part of the function
    full_layers = [out_size, 120, 84, 10]
    shapes      = zip(full_layers[:-1], full_layers[1:])
    full_slices = []
    offset      = num_conv_params
    for shp in shapes:
        num_weights = np.prod(shp) + shp[1]
        full_slices.append(slice(offset, offset + num_weights))
        offset += num_weights
    num_params = offset

    def unpack_full(params):
        for s, shp in zip(full_slices,shapes):
            Wb = np.reshape(params[s], (shp[0]+1, shp[1]))
            yield Wb[:-1, :], Wb[-1,:]

    # prediction function - combine conv MLP and regular MLP
    def predict(params, inputs):
        # conv step
        conv_params = params[:num_conv_params]
        active_out = mlp(inputs, conv_params)

        # fully connected layer step
        active_in = active_out
        for W, b in unpack_full(params):
            active_out = np.dot(active_in, W) + b
            active_in  = np.tanh(active_out)

        # last nonlinearity is softmax
        return active_out - scpm.logsumexp(active_out, axis=1, keepdims=True)

    def loss(params, inputs, targets):
        pred_dist = predict(params, inputs)
        log_like  = np.sum(targets * pred_dist)
        return -log_like + np.sum(params*params)

    def frac_err(params, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(predict(params, X), axis=1))

    # Initialize weights
    W = np.asarray(.1*np.random.randn(num_params), dtype='float32')
    print loss(W, train_images[:100, :, :, :], train_labels[:100, :])
    print grad(loss)(W, train_images[:100, :, :, :], train_labels[:100, :])
    # quick_grad_check(loss, W, (train_images[:100, :, :, :], train_labels[:100, :]))

    # batch up
    from optimizers import adam, rmsprop, sgd

    # define batched loss
    batch_idxs = make_batches(N_data, batch_size=256)
    def opt_loss(W, i):
        idxs = batch_idxs[i % len(batch_idxs)]
        return loss(W, train_images[idxs], train_labels[idxs])

    # define callback function
    print("    Epoch      |    Train err  |   Test error  ")
    def print_perf(th, i, g):
        if i % 10 == 0:
          train_loss = loss(th, train_images[:1000], train_labels[:1000])
          test_perf  = frac_err(th, test_images[:800, :, :, :], test_labels[:800, :])
          train_perf = frac_err(th, train_images[:1000, :, :, :], train_labels[:1000, :])
          print("{epoch:15} | {train_loss:15} | {train_perf:15} | {test_perf:15} " . \
              format(epoch=i,
                     train_loss = train_loss,
                     train_perf=train_perf,
                     test_perf = test_perf))

    # run sgd with momentum
    W = adam(grad(opt_loss), W, callback=print_perf, num_iters=1000, step_size=.001)


