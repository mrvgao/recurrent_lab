"""
Training with given data to solve problems of recurrent neural networks.

1. Input: \vec{X}:  x_0, x_1, x_2, .. x_n, output: y = x_0
2. Input: \vec{X}:  x_0, x_1, x_2, .. x_n, output: y = mean(X)
3. Input: \vec{X}:  x_0, x_1, x_2, .. x_n, output: \vec{y} = (x_n, x_{n-1}, .., x_0)
4. Input: \vec{X}:  x_0, x_1, x_2, .. x_n, output: \vec{Y} = <(x_n + x_{n-1})/2, .. , (x_0 + x_1) /2>

"""
import tensorflow as tf
import numpy as np


