from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import math
import numpy as np
import sys
sys.path.append("../../")
from common.mcts import MCTS
import scipy
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

def run_mcts(clusters, noise, scaler, MAX_CLUSTERS, NOISE_PARAM, noise_mean, noise_std, action_count=24):
    
    _HCC = namedtuple("HierarchicalCluster", "cluster noise terminal")

    class HierarchicalCluster(_HCC, ABC):

        @staticmethod
        def has_children(cluster, n):
            cluster_idxs = np.where(clusters==cluster)[0]
            return n == np.min(noise[cluster_idxs])
        
        def find_children(self):
            cluster_idxs = np.where(clusters==(self.cluster+1))[0]
            return {
                HierarchicalCluster(cluster=clusters[idx], noise=noise[idx], 
                    terminal=HierarchicalCluster.set_terminal(
                        clusters[idx], noise[idx])) for idx in cluster_idxs
            }

        def find_random_child(self):
            cluster_no = np.random.choice(list(set(np.arange(1,MAX_CLUSTERS+1,1).tolist()) - set([self.cluster])))
            noise_value = np.random.choice(noise[clusters==cluster_no])
            return HierarchicalCluster(cluster=cluster_no, noise=noise_value, 
            terminal=HierarchicalCluster.set_terminal(cluster_no, noise_value))

        @staticmethod
        def set_terminal(cluster, n):
            return (cluster == MAX_CLUSTERS-1) or HierarchicalCluster.has_children(cluster, n)
        
        def is_terminal(self):
            return self.terminal

        def reward(self):
            return (MAX_CLUSTERS - self.cluster) / MAX_CLUSTERS + NOISE_PARAM - \
        scaler.transform(((np.array([self.noise]*action_count) - noise_mean) / noise_std).reshape(1,-1)).flatten()[0]
        
    return HierarchicalCluster

def state_space_model(factors, noises, mel_component_matrix, MAX_CLUSTERS):
    def state_space(state):
        factor = factors[state]
        noise = noises[state]
        mel_component = mel_component_matrix[state]
        
        return mel_component, factor, noise
    
    def non_deterministic_hierarchical_clustering(model, n_clusters=MAX_CLUSTERS):
        factor = factors[model]
        Z = ward(pdist(factor))
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        return clusters
    
    def least_noise_model(model, clusters, scaler, 
        MAX_CLUSTERS, NOISE_PARAM, noise_mean, noise_std,
        env=None):
        noise = noises[model]
        node = run_mcts(clusters, noise, 
        scaler, MAX_CLUSTERS, NOISE_PARAM, 
        noise_mean, noise_std)(cluster=1,noise=noise[0], terminal=False)
        mcts = MCTS(env=env)
        
        while True:
            for i in range(25):
                mcts.do_rollout(node)
            node, score = mcts.choose(node)
            if node.terminal:
                break
        state_selected = np.where((clusters == node.cluster) & (noise == node.noise))[0][0]
        return state_selected, score
    
    def marginal_model(action):
        state = mel_component_matrix[action]
        return state
    
    return state_space, non_deterministic_hierarchical_clustering, least_noise_model, marginal_model
