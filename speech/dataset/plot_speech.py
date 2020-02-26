import matplotlib.pyplot as plt
import numpy as np

def plot_ica_points(feat_actors, ica_mixing_array, idx):
    data = np.concatenate([feat_actors['01'][idx] for actor in feat_actors],axis=1)
    scatter = np.dot(data.T, ica_mixing_array[idx])
    plt.scatter(scatter[:,0], scatter[:,1])