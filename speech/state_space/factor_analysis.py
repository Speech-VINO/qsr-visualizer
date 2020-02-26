import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm
import numpy as np
from model import *
import pickle
import sys
import argparse

scale_var = tf.Variable(1.0, dtype=tf.float64)
mel_calibration = tf.Variable(tf.zeros((1,24,120), dtype=tf.float64), dtype=tf.float64)
noise_prob = tf.Variable(0.2, dtype=tf.float64)

ica_mixing_array = pickle.load(open("../dataset/ica_mixing_array.pkl", "rb"))
noise_cov = pickle.load(open("../dataset/noise_cov.pkl", "rb"))
factors = pickle.load(open("../dataset/factors.pkl", "rb"))
noises = pickle.load(open("../dataset/noises.pkl", "rb"))

def parse_args():

    parser = argparse.ArgumentParser()
	parser.add_argument("--n_iter", type=int, required=True, default=5000, help="No. of iterations")
    parser.add_argument("--learning_rate", required=True, type=float, default=0.001, help="Learning rate")
    parser.add_argument("--filepath", required=True, type=argparse.FileType('r'), default=False, help="ONNX filepath")

    return parser.parse_args()

class FactorAnalysis():

    def __init__(self, noise_prob : np.ndarray, noise_cov : dict(), factors):
        self.noise_prob = noise_prob
        self.noise_cov = noise_cov
        self.factors = factors

    def epsilon(self):
        eps = tf.stack([noise_prob * np.sqrt(n**2) for model,n in self.noise_cov.items()])
        return eps

    def forward(self, z):
        return tf.stack([tf.matmul(self.factors[i], z[i]) for i in range(60)]) + self.epsilon()

if __name__ == "__main__":

    print("Training the factor analysis model")

    args = parse_args()

    famodel = FactorAnalysis(noise_prob, noise_cov, np.array(list(factors.values())).astype(np.float64))
    latent = tf.random.normal((60,24,24), dtype=tf.float64)
    d = famodel.forward(latent)

    lr = args.learning_rate

    y_estimated = np.stack(ica_mixing_array).reshape(60,24,120)
    y = tf.matmul(d, mel_calibration)

    optimizer = tf.train.GradientDescentOptimizer(lr)

    loss = tf.reduce_mean(tf.abs(y_estimated - y))

    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    local = tf.local_variables_initializer()

    n_iter = args.n_iter

    with tf.Session() as sess:
        sess.run(init)
        sess.run(local)
        loss_values = []
        train_data = []
        y_vals = []
        captured = []
        for step in tqdm(range(n_iter)):
            # latent model for 60 states and 24 actors
            _, loss_val, noise_p, mel_c, y_val = sess.run([train, loss, noise_prob, mel_calibration, y])
            if step % 1000 == 0:
                print(loss_val)

    save_onnx(factors, noise_prob, noise_cov, args.filepath)
