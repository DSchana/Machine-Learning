import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

import inception5h

# Image manipulation
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

def printTensors(pb_file):
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # Print operations
    for op in graph.get_operations():
        print(op.name)

# IMAGE MANIPULATION FUNCTIONS

def loadImage(filename):
    '''
    Load image with PIL
    '''
    try:
        original = Image.open(filename)
        print("Size of the image is:", original.format, original.size)
        return np.float32(original)
    except:
        print("Unable to load image")

def saveImage(image, filename):
    # Ensure the pixel values are between 0 and 255
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes
    image = image.astype(np.uint8)

    with open(filename, "wb") as file:
        Image.fromarray(image).save(file, "jpeg")

def plotImage(image):
    # Convert the pixel-values to the range between 0.0 and 1.0
    image = np.clip(image / 255.0, 0.0, 1.0)

    # Plot using matplotlib.
    plt.imshow(image, interpolation='lanczos')
    plt.show()

def normalizeImage(x):
    '''
    Noralize image so its values are [0.0, 1.0]. Useful for plotting gradient.
    '''

    # Get the min and max values for all pixels in the input
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm

def plotGradient(gradient):
    '''
    Plot normalize and plot gradient
    '''
    # Normalize gradient to be [0.0, 1.0]
    norm_gradient = normalizeImage(gradient)

    plt.imshow(norm_gradient, interpolation='bilinear')
    plt.show()

def resizeImage(image, size=None, factor=None):
    '''
    Resize image to specified size or scale by specified factor

    :param image:  Image to resize
    :param size:   Desired final size of image
    :param factor: Desired scale factor by which to modify the image
    :return: Resized image
    '''

    if factor is not None:
        # Scale the numpy array for width and height
        size = np.array(image.shape[0:2]) * factor
        size = size.astype(int)
    else:
        size = size[0:2]

    # Numpy and PIL use revered height and width
    size = tuple(reversed(size))

    img = np.clip(image, 0.0, 255.0)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)

    image_resized = img.resize(size, Image.LANCZOS)
    image_resized = np.float32(image_resized)

    return image_resized

# DEEP DREAM ALGORITHM

def getTileSize(num_pixels, tile_size=400):
    '''
    Determine tile size to pan through the full image

    :param num_pixels: Number of pixels in a dimension of the image
    :param tile_size:  Desired tile size
    :return: The tile size
    '''

    # How many times can we repeat a tile of the desired size
    num_tiles = round(num_pixels / tile_size)
    num_tiles = max(1, num_tiles)
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size

def tiledGradient(gradient, image, tile_size=400):
    '''
    Computer the gradient for the input image. The image is split into tiles
    and the gradient for each tile is computed separately. The tiles are chosen
    at random to reduce the amount of visible seams in the final image

    :param gradient:
    :param image:
    :param tile:
    :param size:
    :return:
    '''

    # Allocate an array for the gradient of the entire image
    grad = np.zeros_like(image)

    x_max, y_max, _ = image.shape

    x_tile_size = getTileSize(x_max, tile_size)
    y_tile_size = getTileSize(y_max, tile_size)

    # Random start position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile_size.
    # This is so the border tiles are at least 1/4 of the tile_size,
    # otherwise the tiles may be too small which creates noisy gradients
    x_start = random.randint((-3 / 4) * x_tile_size, (-1 / 4) * y_tile_size)

    while x_start < x_max:
        # End position for the current tile
        x_end = x_start + x_tile_size

        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        y_start = random.randint((-3 / 4) * y_tile_size, (-1 / 4) * y_tile_size)

        while y_start < y_max:
            # End position for the current tile
            y_end = y_start + y_tile_size

            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image tile
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]

            # Creat a feed dict with the image tile
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient value
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile.
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g

            # Advance the start position for the y-axis
            y_start = y_end

        # Advance the start position for the x-axis
        x_start = x_end

    return grad

def optimizeImage(layer_tensor, image, num_iterations=10, step_size=3.0, tile_size=400, show_gradient=False):
    '''
    Use gradient ascent to optimize an imafe so it maximizes the mean
    value of the given layer_tensor

    :param layer_tensor:   Reference to a tensor that will be maxed
    :param image:          Input image used as starting point
    :param num_iterations: Number of optimization iterations to perform
    :param step_size:      Scale for each step of the gradient ascent
    :param tile_size:      Size of the tiles when calculating the gradient
    :param show_gradient:  Plot gradient for each iteration
    :return:
    '''

    # Image copy
    img = image.copy()

    print("Processing image:")

    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math expression to the graph each time this function is called.
    # RIP RAM
    gradient = model.get_gradient(layer_tensor)

    for i in range(num_iterations):
        # Calculate the value of the gradient
        grad = tiledGradient(gradient, img)

        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth_1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth_2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth_3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = grad_smooth_1 + grad_smooth_2 + grad_smooth_3

        # Scane the step size according to the gradient values
        step_size_scaled = step_size / (np.std(grad + 1e-8))

        img += grad * step_size_scaled

        if show_gradient:
            msg = "Gradient min: {0:>9.6f}, max: {1:>9,6f, stepsize: {2:9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))
            plotGradient(grad)
        else:
            # Progress indicator
            print('#', end='')

    return img

def recursiveOptimize(layer_tensor, image, num_repeats=4, rescale_factor=0.7, blend=0.2, num_iterations=10, step_size=3.0, tile_size=400):
    '''
    Recursively blur and downscale the input image. Each downscaled image is run through
    the optimizeImage() function to amplify the patterns that the Inception model sees.

    :param layer_tensor:
    :param image:
    :param num_repeats:
    :param rescale_factor:
    :param blend:
    :param num_iterations:
    :param step_size:
    :param tile_size:
    :return:
    '''

    if (num_repeats > 0):
        # Blur the input image.
        # NOTE: Colout channel is not blurred
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=sigma)#(sigma, sigma, 0.0))

        # Recursively downscale and blur
        img_downscale = resizeImage(img_blur, factor=rescale_factor)
        img_result = recursiveOptimize(layer_tensor, img_downscale, num_repeats - 1, rescale_factor, blend, num_iterations, step_size, tile_size)
        img_upscaled = resizeImage(img_result, size=image.shape)

        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    # Process using DeepDream algo
    img_result = optimizeImage(layer_tensor, image, num_iterations, step_size, tile_size)

    return img_result

print("Ready to start tripping")

inception5h.maybe_download()

model = inception5h.Inception5h()

# TensorFlow Session
session = tf.InteractiveSession(graph=model.graph)

#printTensors("inception/5h/tensorflow_inception_graph.pb")

image = loadImage("image/py_bun treehouse.jpeg")
#plotImage(image)

layer_tensor = model.layer_tensors[2]
layer_tensor

img_results = recursiveOptimize(layer_tensor, image, num_iterations=10, step_size=3.0, rescale_factor=0.7, num_repeats=4, blend=0.2)

saveImage(img_results, "images/output py_bun treehouse.jpeg")
