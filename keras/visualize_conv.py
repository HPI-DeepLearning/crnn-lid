# Adopted from https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py

from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import argparse
import time
from math import ceil, sqrt
from keras.models import load_model
from keras import backend as K
from keras.backend import set_learning_phase

set_learning_phase(0)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == "th":
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def visualize_conv_filters(layer_name, num_filters, input_img, output, img_width, img_height):

    kept_filters = []
    for filter_index in range(0, num_filters):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print("Processing filter %d" % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = output
        if K.image_dim_ordering() == "th":
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.0

        # we start from a gray image with some random noise
        if K.image_dim_ordering() == "th":
            input_img_data = np.random.random((1, 1, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_height, img_width, 1))
        # input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print("Current loss value:", loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print("Filter %d processed in %ds" % (filter_index, end_time - start_time))

    # we will stich the best 64 filters on a 8 x 8 grid.
    n = int(ceil(sqrt(num_filters)))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    #kept_filters = kept_filters[:n * n]
    n = int(ceil(sqrt(len(kept_filters))))
    #
    # import pickle
    # # pickle.dump(kept_filters, open("filters.pickle", "wb"))
    # kept_filters = pickle.load(open("filters.pickle", "rb"))

    remaining_filters = n*n - len(kept_filters)
    for i in range(remaining_filters):
        kept_filters.append((np.zeros((img_height, img_width, 1)), 0.0))

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 1))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]

            # swap X > Y Axis
            img = np.transpose(img, [1, 0, 2])
            # imsave("{}.png".format(i * n + j), np.squeeze(img))

            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    imsave("{0}_{1}x{1}.png".format(layer_name, n), np.transpose(np.squeeze(stitched_filters)))


def visualize_conv_layers(cli_args):

    # dimensions of the generated pictures for each filter.
    img_width = cli_args.width
    img_height = cli_args.height

    model = load_model(cli_args.model_dir)
    model.summary()

    # this is the placeholder for the input images
    input_img = model.input

    for layer in  model.layers:

        if layer.name.startswith("conv"):

            num_filters = layer.output_shape[3]
            visualize_conv_filters(layer.name, num_filters, input_img, layer.output, img_width, img_height)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_dir", required=True)
    parser.add_argument("--layer", dest="layer_name", default="convolution2d_1")
    parser.add_argument('--width', dest='width', default=500, type=int)
    parser.add_argument('--height', dest='height', default=129, type=int)
    parser.add_argument('--filter', dest='num_filter', default=200, type=int)
    cli_args = parser.parse_args()

    visualize_conv_layers(cli_args)