import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_probability as tfp
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm

tfd = tfp.distributions
units_var = tf.Variable(tf.ones(stats.episode_scores.shape[0]))

def units_per_cell():
    return units_var

def beta(min=1e-3, max=0.999, count=500):
    alpha = 0.5
    beta = 0.5
    dist = scipy.stats.beta(alpha, beta)
    return dist, dist.pdf(np.linspace(min,max,count))

def cauchy(min=1e-3, max=0.999, count=500):
    dist = scipy.stats.cauchy()
    return dist, dist.pdf(np.linspace(min,max,count))

def gamma(min=1e-3, max=0.999, count=500):
    a = 0.5
    dist = scipy.stats.gamma(a)
    return dist, dist.pdf(np.linspace(min,max,count))

def rayleigh(min=1e-3, max=0.999, count=500):
    dist = scipy.stats.rayleigh()
    return dist, dist.pdf(np.linspace(min,max,count))

def weibull(min=1e-3, max=0.999, count=500):
    a = 0.5
    c = 0.5
    dist = scipy.stats.exponweib(a, c)
    return dist, dist.pdf(np.linspace(min,max,count))

def execute(stats, distribution, n_iter=200000, lr=1e-1, step_size=10000):

    if distribution == "beta":
        dist, pdf = beta()
    elif distribution == "cauchy":
        dist, pdf = cauchy()
    elif distribution == "gamma":
        dist, pdf = gamma()
    elif distribution == "rayleigh":
        dist, pdf = rayleigh()
    elif distribution == "weibull":
        dist, pdf = weibull()
    else:
        print("The distribution must be specified")
        exit()
    
    x = tf.Variable(tf.ones(stats.episode_scores.shape[0]), shape=stats.episode_scores.shape)
    scaler = MinMaxScaler(feature_range=(0,1)).fit(stats.episode_scores.reshape(-1,1))
    scaled_scores = scaler.transform(stats.episode_scores.reshape(-1,1)).flatten()
    scaled_scores.sort()

    optimizer = tf.train.GradientDescentOptimizer(lr)

    y = MinMaxScaler(feature_range=(0,1)).fit_transform(pdf).reshape(-1,1)).flatten()
    y.sort()
    y_estimated = scaled_scores * units_per_cell()

    loss = tf.reduce_mean(tf.abs(y_estimated - y))

    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    local = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(local)
        loss_values = []
        train_data = []
        y_vals = []
        captured = []
        for step in tqdm(range(n_iter)):
            _, loss_val, units_val, y_val = sess.run([train, loss, units_var, y_estimated])
            loss_values.append(loss_val)
            y_vals.append(y_val)
            if step % step_size == 0:
                print(step, loss_val)

    return pdf, scaled_scores, units_val, y_val, loss_val, loss_values, y_vals
