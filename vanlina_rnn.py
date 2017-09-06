"""
The simplest RNN is

    y_t = X_t + y_{t-1}  if t >= 1
          X_t            if t = 0

But this cannot use gradient descent to optimize the function's performance. Therefore, we
Should add a Variable, which could adjust their values. Based on the 'training' batches,
we could make the function better and better.

Then we get:

    y_t = X_t * W_xy + y_{t_1} + b   if t >= 1
          X_t * W_xy + b             if t == 0

But if our Y's shape is not same as X_t * X_xy, we should add another convert.

Then we get:

    y_t = X_t * W_xy + y_{t_1} + b  if t >= 1
          X_t * W_xy + b            if t == 1

    Y_t = y_t * W_yy + b_hy

We denote y_t as `hidden state` of time t. Then, we change the formula into:

    h_t = X_t * W_hx + h_{t-1} + b if t >= 1
          X_t * W_hx + b           if t == 1

    y_t = h_t * W_yh + b_hy

However, this formula has a disadvantage, each h_t uses all the h_{t-1} equally important. This will make
the result of sequence predict not very well. Because some information will be more important than other information.
Or, some specific time-point information will be more important than some other time-point information.

Therefore, we could change the formula to:

    h_t = X_t * W_hx + h_{t-1} * W_hh + b_hx       if t >= 1
          X_t * W_hx                               if t == 1

    y_t = h_t * W_hy + b_hy

If we need the on-linear convert, we could add no-linear function into this.

    h_t = \sigmoid (X_t * W_hx + h_{t-1} * W_hh + b_hx)  if t >= 1
          \sigmoid (X_t * W_hx)                          if t == 0

    y_t = \sigmoid (h_t * W_hy + b_hy)

"""

from datetime import datetime
import tensorflow as tf
from tensorflow.contrib import rnn
from data_genrator import get_one_batch_trainset

n_features = 1
hidden_neurons = 2
times = 10
y_dimension = 1


# train_X = [
#            [[1], [2], [3]],   # batch0-time 0: [1], time1: [2], time2: [3]
#            [[2], [1], [3]],   # batch1-time 0: [2], time1: [1], time3: [3]
#           ]

# train_y = [
#            [1], # batch0-y = 1,
#            [2], # batch1-y = 2
#           ]

train_X = tf.placeholder(tf.float32, shape=(None, times, n_features), name='X')
train_y = tf.placeholder(tf.float32, shape=(None, y_dimension), name='y')

# gru_cell = rnn.GRUCell(num_units=hidden_neurons)
# cell = rnn.BasicLSTMCell(num_units=hidden_neurons, state_is_tuple=False)
cell = rnn.BasicRNNCell(num_units=hidden_neurons)
# mutilpy_layer_cell = rnn.MultiRNNCell([cell] * 3)

outputs, states = tf.nn.dynamic_rnn(cell, train_X, dtype=tf.float32)

# outputs size is [None, time-steps, neurons]
# state size is [None, neurons]

stacked_outputs = tf.reshape(outputs, [-1, times * hidden_neurons])
# stacked_outputs shape is [None, 200]

W = tf.Variable(tf.truncated_normal(shape=(times * hidden_neurons, y_dimension), stddev=0.05))
b = tf.Variable(tf.zeros(shape=(y_dimension, )))

y_hat = tf.add(tf.matmul(stacked_outputs, W), b, name='y_hat')

loss = tf.losses.mean_squared_error(labels=train_y, predictions=y_hat)

learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

train_epoch = 100

init = tf.global_variables_initializer()

tf.summary.scalar('mse_loss', loss)
merged_summary = tf.summary.merge_all()


def train_epochs(sess):
    sess.run(init)
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_dir = 'tf-logs'
    summary_writer = tf.summary.FileWriter('{}/run-{}'.format(root_dir, now), graph=sess.graph)
    total_train_size = 10000
    batch_size = 128

    total_steps = 0

    saver = tf.train.Saver()

    batch_num = total_train_size // batch_size
    for epoch in range(train_epoch):
        for batch in range(batch_num):
            x, y = get_one_batch_trainset(times_length=times, x_features=n_features, batch_size=batch_num)
            feed_dict = {train_X: x, train_y: y}
            _loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            if batch % 50 == 0:
                print('epoch: {}--batch: {}---loss:{}'.format(epoch, batch, _loss))
                summary = sess.run(merged_summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=total_steps)

                validation_x, validation_y = get_one_batch_trainset(times_length=times,
                                                                    x_features=n_features,
                                                                    batch_size=batch_size)

                val_loss = sess.run([loss], feed_dict={train_X: validation_x, train_y: validation_y})
                print('val loss: {}'.format(val_loss))

            total_steps += 1

    saver.save(sess, 'model/mean_model_length_10', global_step=total_steps)
    print('save model succeed!')


def main(unused_argv):
    with tf.Session() as sess:
        train_epochs(sess)


if __name__ == '__main__':
    tf.app.run()


