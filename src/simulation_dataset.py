import numpy as np
import os
import gzip
import random
from numpy.random import RandomState

class MovingMNIST2:
    """The MovingMNIST dataset with missing values, the goal is to impute the missing frames"""
    def __init__(self,
                 root,
                 n_frames,
                 mask,
                 num_digits,
                 image_size,
                 digit_size,
                 N,
                 transform=None,
                 use_fixed_dataset=False,
                 random_state=None):
        '''if use_fixed_dataset = True, the mnist_test_seq.npy in the root folder will be loaded'''
        super().__init__()
        self.use_fixed_dataset = use_fixed_dataset
        if not use_fixed_dataset:
            self.mnist = self.load_mnist(root, image_size=digit_size)
        else:
            self.dataset = self.load_fixed_set(root)

            # take a slice
            assert (self.dataset.shape[1] > N)
            self.dataset = self.dataset[:, :N, ...]

        self.length = N
        self.n_frames = n_frames
        self.mask = mask
        self.transform = transform
        # For generating data
        self.image_size_ = image_size
        self.digit_size_ = digit_size
        self.step_length_ = 0.1
        self.num_digits = num_digits
        self.random_state = random_state
        if random_state is not None:
            self.rng = RandomState(random_state)
        else:
            self.rng = RandomState(random.randint(1, 1e4))
    def load_mnist(self, root, image_size):
        # Load MNIST dataset for generating training data.
        path = os.path.join(root, 'train-images-idx3-ubyte.gz')
        with gzip.open(path, 'rb') as f:
            mnist = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist = mnist.reshape(-1, image_size, image_size)
        return mnist

    def load_fixed_set(self, root):
        # Load the fixed dataset
        filename = 'mnist_test_seq.npy'
        path = os.path.join(root, filename)
        dataset = np.load(path)
        dataset = dataset[..., np.newaxis]
        return dataset

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_



        x = self.rng.random()
        y = self.rng.random()
        theta = self.rng.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros(
            (self.n_frames, self.image_size_, self.image_size_),
            dtype=np.float32)
        for n in range(self.num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames)
            ind = self.rng.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom,
                left:right] = np.maximum(data[i, top:bottom, left:right],
                                         digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames
        mask = np.array(self.mask)  # the mask for the idx th sample


        # Sample number of objects
        # Generate data on the fly
        if not self.use_fixed_dataset:
            images = self.generate_moving_mnist()
        else:
            images = self.dataset[:, idx, ...]
            images = np.float32(images)

        # if self.transform is not None:
        #     images = self.transform(images)

        r = 1
        w = int(self.image_size_ / r)
        # w = int(64 / r)
        images = images.reshape(
            (length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape(
            (length, r * r, w, w))


        images = images / 255
        images_input = images[mask == 1, ...]
        images_true = images[1:, ...]
        actions = []
        state = []
        return [idx, mask, images_input, actions, state, images_true]

    def __len__(self):
        return self.length



