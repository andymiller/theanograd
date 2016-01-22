import autograd.numpy as np
import autograd.scipy.misc as scpm
from autograd import grad

from theanograd.util import load_mnist, make_batches
import theanograd.conv_mlp as cmlp
from theanograd.conv_mlp import make_conv_mlp, conv_layer

import theano.tensor as T

if __name__=="__main__":
    #load in mnist data
    if 'train_images' not in locals():
        train_images, train_labels, test_images, test_labels = load_mnist() #data
        N_data       = train_images.shape[0]

    ####################################################
    # define a convolutional multi-layer perceptron    #
    ####################################################
    L2_reg = 1.0
    input_shape = (1, 28, 28)
    layer_specs = [
        conv_layer(filter_shape = (5, 5), num_filters=6, pool_shape=(2,2), nonlinearity=T.nnet.relu),
        conv_layer(filter_shape = (5, 5), num_filters=16, pool_shape=(2,2), nonlinearity=T.nnet.relu)
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
    from theanograd.optimizers import adam, rmsprop, sgd

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
    W = adam(grad(opt_loss), W, callback=print_perf, num_iters=20000, step_size=.01)


