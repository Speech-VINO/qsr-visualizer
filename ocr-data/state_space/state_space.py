from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import math
import numpy as np
import sys
sys.path.append("../../common")
from mcts import MCTS
import scipy
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

def run_mcts(clusters, similarity, scaler, MAX_CLUSTERS, NOISE_PARAM, similarity_mean, similarity_std, action_count):
    
    _HCC = namedtuple("HierarchicalCluster", "idx cluster similarity terminal")

    class HierarchicalCluster(_HCC, ABC):

        @staticmethod
        def has_children(idx, cluster, n):
            cluster_idxs = np.where(clusters==cluster)[0]
            l = list(set(similarity[idx][cluster_idxs].tolist()) - set([1]))
            return True if len(l) == 0 else n == max(l)
        
        def find_children(self):
            cluster_idxs = np.where(clusters==(self.cluster+1))[0]
            return {
                HierarchicalCluster(cluster=clusters[idx], similarity=similarity[self.idx][idx], idx=idx, 
                    terminal=HierarchicalCluster.set_terminal(idx,
                        clusters[idx], similarity[self.idx][idx])) for idx in cluster_idxs
            }

        def find_random_child(self):
            cluster_no = np.random.choice(
                list(set(np.arange(1,MAX_CLUSTERS+1,1).tolist()) - set([self.cluster]))
            )
            sim_value = np.random.choice(similarity[self.idx][clusters==cluster_no])
            idx = np.random.choice(np.where(clusters==cluster_no)[0])
            return HierarchicalCluster(cluster=cluster_no, similarity=sim_value, idx=idx, 
            terminal=HierarchicalCluster.set_terminal(idx, cluster_no, sim_value))

        @staticmethod
        def set_terminal(idx, cluster, n):
            return (cluster == MAX_CLUSTERS-1) or HierarchicalCluster.has_children(idx, cluster, n)
        
        def is_terminal(self):
            return self.terminal

        def reward(self):
            return (MAX_CLUSTERS - self.cluster) / MAX_CLUSTERS + NOISE_PARAM - \
        scaler.transform(((np.array([self.similarity]*action_count) - similarity_mean) / similarity_std).reshape(1,-1)).flatten()[0]
        
    return HierarchicalCluster

def state_space_model(similarity, tfidf, MAX_CLUSTERS, action_count):
    def state_space(state):
        sim = similarity[state]
        
        return sim
    
    def non_deterministic_hierarchical_clustering(model, n_clusters=MAX_CLUSTERS):
        sim = similarity[model].reshape(-1,1)
        Z = ward(pdist(sim))
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        return clusters
    
    def maximum_similarity_model(model, clusters, 
        scaler, MAX_CLUSTERS, NOISE_PARAM, similarity_mean, similarity_std, env=None):
        sim = similarity[model]
        node = run_mcts(clusters, similarity, scaler, MAX_CLUSTERS, 
        NOISE_PARAM, similarity_mean, similarity_std, action_count)(idx=0,cluster=1,similarity=sim[0], terminal=False)
        mcts = MCTS(env=env)
        
        while True:
            for i in range(25):
                mcts.do_rollout(node)
            node, score = mcts.choose(node)
            if node.terminal:
                break
        idxs = np.where((similarity == node.similarity))
        idxs = np.where((clusters[idxs[0]] == node.cluster))[0]
        state_selected = idxs[0]
        return state_selected, score
    
    def marginal_model(action):
        state = tfidf.todense()
        
        return state
    
    return state_space, non_deterministic_hierarchical_clustering, maximum_similarity_model, marginal_model
