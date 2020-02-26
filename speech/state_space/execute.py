from lib.running_variance import RunningVariance
from itertools import count
from lib import plotting
from lib.envs.bandit import BanditEnv, ActionSpace
import numpy as np
import sys
import pickle
from state_space import *
import argparse
import random_distributions
import onnxruntime as nxrun
from sklearn.preprocessing import MinMaxScaler
import os
import onnxruntime as nxrun

def parse_args():

    parser = argparse.ArgumentParser()
	parser.add_argument("--distribution", type=str, default=False, help="Distribution")

    return parser.parse_args()

sess = nxrun.InferenceSession("./models/factor_analysis.onnx")

latent = np.load('../dataset/latent.npy')
mel_component_matrix = sess.run(None, {'latent': latent})
mel_component_matrix = mel_component_matrix[0]
factors = pickle.load(open("../dataset/factors.pkl", "rb"))
noises = pickle.load(open("../dataset/noises.pkl", "rb"))

state_dim = 24 # Dimension of state space
action_count = 60 # Number of actions
hidden_size = 256 # Number of hidden units
update_frequency = 20

noise_values = np.array(list(noises.values()))
noise_mean = np.mean(noise_values)
noise_std = np.std(noise_values)

scaler = MinMaxScaler(feature_range=(0,6)).fit((noise_values - noise_mean)/noise_std)

state_space, non_deterministic_hierarchical_clustering, least_noise_model, marginal_model = \
state_space_model(factors, noises, mel_component_matrix)

class PGCREnv(BanditEnv):
    def __init__(self, num_actions = 10, 
    observation_space = None, distribution = "factor_model", evaluation_seed=387):
        super(BanditEnv, self).__init__()
        
        self.action_space = ActionSpace(range(num_actions))
        self.distribution = distribution
        
        self.observation_space = observation_space
        
        np.random.seed(evaluation_seed)
        
        self.reward_parameters = None
        if distribution == "bernoulli":
            self.reward_parameters = np.random.rand(num_actions)
        elif distribution == "normal":
            self.reward_parameters = (np.random.randn(num_actions), np.random.rand(num_actions))
        elif distribution == "heavy-tail":
            self.reward_parameters = np.random.rand(num_actions)
        elif distribution == "factor_model":
            self.reward_parameters = (np.array(list(factors.values())).sum(axis=2), 
                          np.array(list(noises.values())))
        else:
            print("Please use a supported reward distribution", flush = True)
            sys.exit(0)
        
        if distribution != "normal":
            self.optimal_arm = np.argmax(self.reward_parameters)
        else:
            self.optimal_arm = np.argmax(self.reward_parameters[0])
    
    def reset(self):
        self.is_reset = True
        action = np.random.randint(0,action_count)
        return mel_component_matrix[action], list(factors.keys())[action]
    
    def compute_gap(self, action):
        if self.distribution == "factor_model":
            gap = np.absolute(self.reward_parameters[0][self.optimal_arm] - self.reward_parameters[0][action])
        elif self.distribution != "normal":
            gap = np.absolute(self.reward_parameters[self.optimal_arm] - self.reward_parameters[action])
        else:
            gap = np.absolute(self.reward_parameters[0][self.optimal_arm] - self.reward_parameters[0][action])
        return gap
    
    def step(self, action):
        self.is_reset = False
        
        valid_action = True
        if (action is None or action < 0 or action >= self.action_space.n):
            print("Algorithm chose an invalid action; reset reward to -inf", flush = True)
            reward = float("-inf")
            gap = float("inf")
            valid_action = False
        
        if self.distribution == "bernoulli":
            if valid_action:
                reward = np.random.binomial(1, self.reward_parameters[action])
                gap = self.reward_parameters[self.optimal_arm] - self.reward_parameters[action]
        elif self.distribution == "normal":
            if valid_action:
                reward = self.reward_parameters[0][action] + self.reward_parameters[1][action] * np.random.randn()
                gap = self.reward_parameters[0][self.optimal_arm] - self.reward_parameters[0][action]
        elif self.distribution == "heavy-tail":
            if valid_action:
                reward = self.reward_parameters[action] + np.random.standard_cauchy()
                gap = self.reward_parameters[self.optimal_arm] - self.reward_parameters[action]        #HACK to compute expected gap
        elif self.distribution == "factor_model":
            if valid_action:
                reward = np.linalg.norm(
                    self.reward_parameters[0][action],1
                ) + \
                np.linalg.norm(
                    self.reward_parameters[1][action],1
                ) * np.random.randn()
        else:
            print("Please use a supported reward distribution", flush = True)
            sys.exit(0)
            
        observation = marginal_model(action)
        
        return(observation, list(factors.keys())[action], reward, self.is_reset, '')

def discount_rewards(r, factor_model, gamma=0.999):
    """Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r,dtype=np.float32)
    running_add = 0
    f_ = 0
    f = np.linalg.norm(factor_model[0],1)
    running_add = f
    for t in reversed(range(0, r.size)):
        if (t < (r.size - 1) and r.size >= 2):
            f_ = np.linalg.norm(factor_model[t+1],1)
            f = np.linalg.norm(factor_model[t],1)
            running_add = running_add + gamma * f_ - f
        running_add = running_add + r[t]
        discounted_r[t] = running_add
    return discounted_r

def execute(MAX_CLUSTERS=11, num_epochs=2, max_number_of_episodes=500, reward_sum=0):

    running_variance = RunningVariance()

    epoch_stats = []
    net_actions = []
    net_rewards = []
    net_scores = []

    for epoch in tqdm(range(num_epochs)):
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(max_number_of_episodes),
            episode_rewards=np.zeros(max_number_of_episodes),
            episode_running_variance=np.zeros(max_number_of_episodes),
            episode_scores=np.zeros(max_number_of_episodes),
            losses=np.zeros(max_number_of_episodes))

        env = PGCREnv(num_actions = action_count, observation_space = np.zeros((state_dim,1)))
        
        for episode_number in tqdm(range(max_number_of_episodes)):
            states, rewards, labels, scores = [],[],[],[]
            done = False

            observation, model = env.reset()
            factor_sequence = []
            t = 1
            for state_dim_i in range(state_dim):
                done = False
                while not done:
                    
                    state = np.ascontiguousarray(np.reshape(observation[:,state_dim_i], [1,120]).astype(np.float32))
                    states.append(state)

                    is_reset = False
                    score = 0.0
                    action, score = least_noise_model(
                        model, non_deterministic_hierarchical_clustering(model), 
                        scaler, MAX_CLUSTERS, NOISE_PARAM, noise_mean, noise_std, 
                        env=env
                    )
                    is_reset = env.is_reset
                    
                    net_actions.append(action)

                    z = np.ones((1,state_dim)).astype(np.float32) * 1.25/120
                    z[:,state_dim_i] = 0.75
                    labels.append(z)
                    
                    # step the environment and get new measurements
                    observation, model, reward, done, _ = env.step(action)
                    
                    done = is_reset if is_reset is True else False

                    observation = np.ascontiguousarray(observation)

                    net_rewards.append(reward)
                    net_scores.append(score)

                    reward_sum += float(reward)

                    # Record reward (has to be done after we call step() to get reward for previous action)
                    rewards.append(float(reward))

                    factor_sequence.append(list(factors.values())[action])

                    stats.episode_rewards[episode_number] += reward
                    stats.episode_lengths[episode_number] = t
                    stats.episode_scores[episode_number] += score

                    t += 1

            # Stack together all inputs, hidden states, action gradients, and rewards for this episode
            epr = np.vstack(rewards).astype(np.float32)

            # Compute the discounted reward backwards through time.
            discounted_epr = discount_rewards(epr, factor_sequence)
            
            for discounted_reward in discounted_epr:
                # Keep a running estimate over the variance of of the discounted rewards
                running_variance.add(discounted_reward.sum())

            stats.episode_running_variance[episode_number] = running_variance.get_variance()
        
        plotting.plot_pgresults(stats)
        epoch_stats.append(stats)

    return stats

if __name__ == "__main__":

    print("Training the state space with MCTS simulation and PGCR Env")

    if not args.distribution:
        print("The distribution must be specified, or specify all for executing all distributions: \
            (beta, cauchy, gamma, rayleigh, weibull)")
        exit()
    
    if not os.path.isfile('tmp_models/stats.npy'):
        args = parse_args()

        stats = execute()

        np.save('tmp_models/stats.npy', stats)
    else:
        print("Using Trained statistical data...")
        stats = np.load('tmp_models/stats.npy')
    
    print("Training the distributions with obtained reward scores from MCTS..")

    if args.distribution != "all":
        z = args.distribution
        print("""Running tensorflow model for '{z}' distribution""".format(z=z))

        pdf, scaled_scores, units_val, y_val, loss_val, loss_values, y_vals = \
            random_distributions.execute(stats, distribution)
        np.save("""tmp_models/{dist}.npy""".format(dist=args.distribution), pdf)
        np.save("""tmp_models/{units}.npy""".format(units=args.distribution+"_units"), units_val)
        np.save("""tmp_models/{score}.npy""".format(score=args.distribution+"_scores"), scaled_scores)
        np.save("""tmp_models/{timestamp}.npy""".format(timestamp=args.distribution+"_timestamp"), stats.episode_scores.argsort())
    else:
        for z in ('beta', 'cauchy', 'gamma', 'rayleigh', 'weibull'):
            
            print("""Running tensorflow model for '{z}' distribution""".format(z=z))

            pdf, scaled_scores, units_val, y_val, loss_val, loss_values, y_vals = \
                random_distributions.execute(stats, z)

            np.save("""tmp_models/{dist}.npy""".format(dist=args.distribution), pdf)
            np.save("""tmp_models/{units}.npy""".format(units=args.distribution+"_units"), units_val)
            np.save("""tmp_models/{score}.npy""".format(score=args.distribution+"_scores"), scaled_scores)
            np.save("""tmp_models/{timestamp}.npy""".format(timestamp=args.distribution+"_timestamp"), stats.episode_scores.argsort())