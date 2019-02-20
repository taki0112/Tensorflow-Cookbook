import tensorflow as tf
from scipy import misc
import numpy as np
import random

class ImageData:

    def __init__(self, load_size, channels, augment_flag):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = img/127.5 - 1 # -1 ~ 1

    return img

def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizer_fully(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully

def tf_rgb_to_gray(x) :
    x = (x + 1.0) * 0.5
    x = tf.image.rgb_to_grayscale(x)

    x = (x * 2) - 1.0

    return x

def RGB2LAB(srgb):
    srgb = inverse_transform(srgb)

    lab = rgb_to_lab(srgb)
    l, a, b = preprocess_lab(lab)

    l = tf.expand_dims(l, axis=-1)
    a = tf.expand_dims(a, axis=-1)
    b = tf.expand_dims(b, axis=-1)

    x = tf.concat([l, a, b], axis=-1)

    return x

def LAB2RGB(lab) :
    lab = inverse_transform(lab)

    rgb = lab_to_rgb(lab)
    rgb = tf.clip_by_value(rgb, 0, 1)

    # r, g, b = tf.unstack(rgb, axis=-1)
    # rgb = tf.concat([r,g,b], axis=-1)

    x = (rgb * 2) - 1.0

    return x

def rgb_to_lab(srgb):
    with tf.name_scope('rgb_to_lab'):
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        with tf.name_scope('xyz_to_cielab'):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope('lab_to_rgb'):
        lab_pixels = tf.reshape(lab, [-1, 3])
        with tf.name_scope('cielab_to_xyz'):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope('xyz_to_srgb'):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=-1)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range [-128, 127]
        # [0, 100] => [-1, 1],  ~[-128, 127] => [-1, 1]

        L_chan = L_chan * 255.0 / 100.0
        a_chan = a_chan + 128
        b_chan = b_chan + 128

        L_chan /= 255.0
        a_chan /= 255.0
        b_chan /= 255.0

        L_chan = (L_chan - 0.5) / 0.5
        a_chan = (a_chan - 0.5) / 0.5
        b_chan = (b_chan - 0.5) / 0.5

        return [L_chan, a_chan, b_chan]