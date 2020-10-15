import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow
import random

# Class representation of network model

tf.disable_v2_behavior()

CHECKPOINT_DIR="./Model/Checkpoints2/"
class Model(object):
    # Initialize model
    def __init__(self, x_data, y_data, learning_rate, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialize training dataset
        self.initialize_dataset()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    # Initialize session
    def set_session(self, sess):
        self.sess = sess
    # Initialize dataset
    def initialize_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x_data,self.y_data))
        self.dataset = self.dataset.apply(tensorflow.data.experimental.shuffle_and_repeat(self.batch_size*5))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(self.batch_size*5)
        self.dataset = self.dataset.make_one_shot_iterator()
        self.dataset = self.dataset.get_next()

    # Define graph for model
    def build_model(self):

        # Define placeholders for input and ouput values
        self.x = tf.placeholder(tf.float32, [None, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')

        # Define placeholder for learning rate
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')

        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(self.x, 10, activation=tf.nn.relu)

        # Define fully-connected layer with 20 hidden units
        h = tf.layers.dense(h, 20, activation=tf.nn.relu)

        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(h, 10, activation=tf.nn.relu)

        # Define fully-connected layer to single ouput prediction
        self.pred = tf.layers.dense(h, 1, activation=None)

        # Define loss function
        self.loss = tf.reduce_mean(tf.pow(self.pred - self.y, 2))

        # Define optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rt) \
                             .minimize(self.loss, global_step=self.global_step)

        # Define summary operation for saving losses
        tf.summary.scalar("Loss", self.loss)
        self.merged_summaries = tf.summary.merge_all()

    # Train model
    def train(self):

        # Define summary writer for saving log files
        self.writer = tf.summary.FileWriter('./Model/logs/', graph=tf.get_default_graph())

        # Iterate through 20000 training steps
        while not self.sess.should_stop():

            # Update globale step
            step = tf.train.global_step(self.sess, self.global_step)

            # Retrieve batch from data loader
            x_batch, y_batch = self.sess.run(self.dataset)

            # Apply decay to learning rate every 1000 steps
            if step % 1000 == 0:
                self.learning_rate = 0.9*self.learning_rate

            # Run optimization operation for current mini-batch
            fd = {self.x: x_batch, self.y: y_batch, self.learning_rt: self.learning_rate}
            self.sess.run(self.optim, feed_dict=fd)

            # Save summary every 100 steps
            if step % 100 == 0:
                summary = self.sess.run(self.merged_summaries, feed_dict=fd)
                self.writer.add_summary(summary, step)
                self.writer.flush()

            # Display progress every 1000 steps
            if step % 1000 == 0:
                print(tf.train.list_variables(tf.train.latest_checkpoint(CHECKPOINT_DIR)))


                loss = self.sess.run(self.loss, feed_dict=fd)
                print("Step %d:  %.10f" %(step,loss))

    # Define method for computing model predictions
    def predict(self, eval_pts):
        return self.sess.run(self.pred, feed_dict={self.x: eval_pts})

    # Evaluate model
    def evaluate(self):
        # Compute final loss on full dataset
        fd = {self.x: self.x_data, self.y: self.y_data}
        final_loss = self.sess.run(self.loss, feed_dict=fd)
        print("FINAL LOSS = %.10f" %(final_loss))

        # Plot predicted and true values for qualitative evaluation
        eval_pts = np.expand_dims(np.linspace(-np.pi/2, np.pi/2, 1000) , 1)
        predictions = self.predict(eval_pts)
        true_values = np.sin(eval_pts)
        plt.plot(eval_pts[:,0], predictions[:,0], 'b')
        plt.plot(eval_pts[:,0], true_values[:,0], 'r')
        plt.show()

DATA_CHECKPOINT_KEY = "data_checkpoint"
TENSOR_DATA_CHECKPOINT_KEY = "data_checkpoint:0"

class DataCheckpointSaverListener(tf.estimator.CheckpointSaverListener):
    def __init__(self):
        pass

    def begin(self):
        ckpt = tf.placeholder(tf.string, name="hello_name")
        var = tf.Variable("invalid invalid", name = DATA_CHECKPOINT_KEY)
        self.tensor = var.assign(ckpt)

    def before_save(self, session, global_step_value):
        print('About to write a checkpoint at step {}'.format(global_step_value))
        # store the dataset checkpoint here
        res = session.run(self.tensor, {"hello_name:0": "{}_{}".format(global_step_value, self._current_dataset())})
        print("afrer run", res)
    def _current_dataset(self):
        """get current dataset checkpoint"""
        seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        sa = []
        for i in range(8):
            sa.append(random.choice(seed))
        return ''.join(sa)


# Initialize and train model 
def main():

    # Create artificial data 
    x_data = np.pi/2 * np.random.normal(size=[100*10000, 1])
    y_data = np.sin(x_data)

    # Specify initial learning rate
    learning_rate = 0.00075

    # Specify training batch size
    batch_size = 100

    # Initialize model
    model = Model(x_data, y_data, learning_rate, batch_size)

    # Specify number of training steps
    training_steps = 20000

    listener = DataCheckpointSaverListener()
    saver_hook = tf.estimator.CheckpointSaverHook(
        checkpoint_dir=CHECKPOINT_DIR,
        save_steps=1000,
        listeners=[listener])

    # Initialize TensorFlow monitored training session
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=CHECKPOINT_DIR,
            is_chief=True,
            hooks = [tf.train.StopAtStepHook(last_step=training_steps)],
            chief_only_hooks=[saver_hook],
            save_summaries_steps = None, save_checkpoint_steps = 2000) as sess:

        ckpt_datablk = tf.get_default_graph().get_tensor_by_name(TENSOR_DATA_CHECKPOINT_KEY)
        print("restore", sess.run(ckpt_datablk))

        # Initialize model session
        model.set_session(sess)

        # Train model
        model.train()


    # Create new session for model evaluation
    with tf.Session() as sess:

        # Restore network parameters from latest checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))

        ckpt_datablk = tf.get_default_graph().get_tensor_by_name(TENSOR_DATA_CHECKPOINT_KEY)
        print("restore when predict", sess.run(ckpt_datablk))

        # Initialize model with new session
        model.set_session(sess)

        # Evaluate model
        model.evaluate()

# Run main() function when called directly
if __name__ == '__main__':
    main()
