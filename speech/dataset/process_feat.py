import sys
sys.path.append("../audio")
import htk_featio as htk
import speech_sigproc as sp
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
import librosa
from sklearn.decomposition import FastICA
import tensorflow as tf
import factor_analysis

def process_feat(samp_rate=48000):
    wav_files = glob("dev/*/*.wav")
    strings = np.unique(pd.Series(wav_files).replace(to_replace=r"\.\.\/[a-z0-9]+\/dev\/[a-zA-Z0-9]+\/", value="", regex=True).values.flatten()).tolist()

    for ii,wav_file in tqdm(enumerate(wav_files)):
        
        x, s = librosa.core.load(wav_file)

        fe = sp.FrontEnd(samp_rate=s,mean_norm_feat=True,
                frame_duration=0.032, frame_shift=0.02274,hi_freq=8000)

        feat = fe.process_utterance(x)
        if strings[ii].split("-")[6].replace(".wav","") not in feat_actors:
            feat_actors[strings[ii].split("-")[6].replace(".wav","")] = []
        feat_actors[strings[ii].split("-")[6].replace(".wav","")].append(feat)

    return feat_actors

def process_ica(feat_actors, random_state=42):
    ica_mixing_array = []
    n_actors = len(feat_actors)
    for i in tqdm(range(len(feat_actors['01']))):
        # for each string
        data = np.concatenate([feat_actors[actor][i] for actor in feat_actors],axis=1)
        ica = FastICA(n_components=n_actors, random_state=42)
        ica.fit(data.T)
        ica_mixing_array.append(ica.mixing_)

def process_factors(ica_mixing_array, strings):
    factors = {}
    noises = {}
    noise_cov = {}

    for state_dim,ica_mixing in tqdm(enumerate(ica_mixing_array)):
        f = factor_analysis.factors.Factor(ica_mixing_array[state_dim], 
    factor_analysis.posterior.Posterior(np.cov(ica_mixing.T), 
    np.concatenate([np.mean(ica_mixing,axis=0).reshape(1,-1)/ica_mixing.shape[1]]*ica_mixing.shape[1],axis=0)))

        noise = factor_analysis.noise.Noise(f, f.posterior)

        with tf.Session() as sess:
            factors[strings[state_dim].split("/")[-1].replace(".wav","")] = f.create_factor().eval()
            noise_cov[strings[state_dim].split("/")[-1].replace(".wav","")] = noise.noise.eval()
            noises[strings[state_dim].split("/")[-1].replace(".wav","")] = noise.create_noise(f.create_factor()).eval()