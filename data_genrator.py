import random
import numpy as np


def get_one_time_point_data(x_features=1):
    _min = random.randint(-100, 0)
    _max = random.randint(0, 100)
    return np.array([random.randint(_min, _max) for _ in range(x_features)])


def get_one_timeserise_data(one_time_step_data_generator, x_features=1, times_length=5):
    x = np.array([one_time_step_data_generator(x_features=x_features) for _ in range(times_length)])

    y = np.array([np.mean(x)])

    return x, y


def get_one_batch_trainset(times_length=5, x_features=1, batch_size=128):
    train_X = []
    train_y = []
    for _ in range(batch_size):
        X, Y = get_one_timeserise_data(get_one_time_point_data,
                                       x_features=x_features,
                                       times_length=times_length)
        train_X.append(X)
        train_y.append(Y)

    return np.array(train_X), np.array(train_y)


one_dataset = get_one_time_point_data(x_features=5)
assert one_dataset.shape == (5, )

one_timeserise_x, one_timeserise_y = get_one_timeserise_data(get_one_time_point_data, x_features=1, times_length=5)

assert one_timeserise_x.shape == (5, 1)
assert one_timeserise_y.shape == (1, ), 'shape is: {}'.format(one_timeserise_y.shape)
assert np.mean(one_timeserise_x) == one_timeserise_y

one_trainset = get_one_batch_trainset(times_length=5, x_features=1, batch_size=128)
x, y = one_trainset

assert x.shape == (128, 5, 1)
assert y.shape == (128, 1)

print('test done!')
