import sys
sys.path.append("../../")
from preprocess import *
from common import lib
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
    parser.add_argument("--MAX_CLUSTERS", type=int, default=1000, help="Max Clusters")
    parser.add_argument("--num_epochs", type=int, default=2, help="No. of epochs")
    parser.add_argument("--max_number_of_episodes", type=int, default=500, help="Max No. of epochs")

    return parser.parse_args()

corpus, labels = obtain_corpus(obtain_files())
tfidf = tfidf_vectorizer(corpus)
similarity, similarity_mean, similarity_std = \
    calculate_similarity(latent_semantic_analysis(tfidf))
one_hot = one_hot_encoded_labels(labels)

state_dim = tfidf.todense().shape[0] # Dimension of state space
action_count = tfidf.todense().shape[1] # Number of actions
update_frequency = 20

class PGCREnv(BanditEnv):
    def __init__(self, num_actions = 10, observation_space = None, distribution = "factor_model", evaluation_seed=387):
        super(BanditEnv, self).__init__()
        
        self.action_space = ActionSpace(range(num_actions))
        self.distribution = distribution
        
        self.observation_space = observation_space
        
        np.random.seed(evaluation_seed)
        
        self.reward_parameters = [similarity.sum(axis=1), similarity.sum(axis=1)]
        
        self.optimal_arm = np.argmax(self.reward_parameters)
    
    def reset(self):
        self.is_reset = True
        action = np.random.randint(0,action_count)
        return tfidf.todense(), action
    
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
                reward = self.reward_parameters[0][action] + \
                self.reward_parameters[1][action] * np.random.randn()
        else:
            print("Please use a supported reward distribution", flush = True)
            sys.exit(0)
            
        observation = marginal_model(action)
        
        return (observation, action, reward, self.is_reset, '')

def discount_rewards(r, svd_model, gamma=0.999):
    """Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r,dtype=np.float32)
    running_add = 0
    f_ = 0
    f = np.linalg.norm(svd_model[0],1)
    running_add = f
    for t in reversed(range(0, r.size)):
        if (t < (r.size - 1) and r.size >= 2):
            f_ = np.linalg.norm(svd_model[t+1],1)
            f = np.linalg.norm(svd_model[t],1)
            running_add = running_add + gamma * f_ - f
        running_add = running_add + r[t]
        discounted_r[t] = running_add
    return discounted_r

def execute(args):
    MAX_CLUSTERS = args.MAX_CLUSTERS
    num_epochs = args.num_epochs
    max_number_of_episodes = args.max_number_of_episodes

    running_variance = RunningVariance()
    reward_sum = 0

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
            for state_dim_i in tqdm(range(state_dim)):
                done = False
                
                while not done:
                    
                    state = np.ascontiguousarray(
                        np.reshape(observation[state_dim_i,:], [1,action_count]).astype(np.float32)
                    )
                    states.append(state)

                    # Run the policy network and get an action to take.
                    # probability with actions
                    is_reset = False
                    score = 0.0
                    action, score = maximum_similarity_model(
                        model, non_deterministic_hierarchical_clustering(model), 
                        scaler, MAX_CLUSTERS, NOISE_PARAM, similarity_mean, similarity_std, 
                        env=env
                    )
                    is_reset = env.is_reset
                    
                    net_actions.append(action)

                    labels.append(one_hot[state_dim_i].values)
                    
                    # step the environment and get new measurements
                    observation, model, reward, done, _ = env.step(action)

                    done = is_reset if is_reset is True else False

                    observation = np.ascontiguousarray(observation)

                    net_rewards.append(reward)
                    net_scores.append(score)

                    reward_sum += float(reward)

                    # Record reward (has to be done after we call step() to get reward for previous action)
                    rewards.append(float(reward))

                    factor_sequence.append(similarity[action])

                    stats.episode_rewards[episode_number] += reward
                    stats.episode_lengths[episode_number] = t
                    stats.episode_scores[episode_number] += score

                    t += 1
                    
            # Compute the discounted reward backwards through time.
            discounted_epr = discount_rewards(epr, factor_sequence)
            
            for discounted_reward in discounted_epr:
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
    
    if not os.path.isfile('tmp_models/episode_scores.npy'):
        print("Training for scores takes time for OCR data")
        args = parse_args()

        stats = execute(args)

        np.save('tmp_models/episode_scores.npy', stats.episode_scores)
    else:
        print("Using Trained statistical data...")
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(max_number_of_episodes),
            episode_rewards=np.zeros(max_number_of_episodes),
            episode_running_variance=np.zeros(max_number_of_episodes),
            episode_scores=np.zeros(max_number_of_episodes),
            losses=np.zeros(max_number_of_episodes))
        episode_scores = np.load('tmp_models/episode_scores.npy')
        stats['episode_scores'] = episode_scores

    
    print("Training the distributions with obtained reward scores from MCTS..")

    if args.distribution != "all":
        z = args.distribution
        print("""Running tensorflow model for '{z}' distribution""".format(z=z))

        pdf, scaled_scores, units_val, y_val, loss_val, loss_values, y_vals = \
            random_distributions.execute(stats, args.distribution)
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