import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", type=str, default=False, help="Distribution")

    return parser.parse_args()

if __name__ == "__main__":

    dirname = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()

    if args.distribution and args.distribution != "all":
        z = args.distribution
        print("""\nPlotting results for '{z}' distribution\n""".format(z=z))

        if os.path.isfile(dirname + """/tmp_models/{score}.npy""".format(score="episode_scores")):
            smoothing_window=20
            episode_scores = np.load(dirname + """/tmp_models/{score}.npy""".format(score="episode_scores"))
            scores_smoothed = pd.Series(episode_scores).rolling(smoothing_window, min_periods=smoothing_window).mean()
            plt.title("""Smoothed episode scores""")
            plt.plot(np.arange(0,len(scores_smoothed)), scores_smoothed, color='green')
            plt.show()

        if os.path.isfile(dirname + """/tmp_models/{score}.npy""".format(score=z+"_scores")):
            scaled_scores = np.load(dirname + """/tmp_models/{score}.npy""".format(score=z+"_scores"))
            idx = scaled_scores.argsort()
            scaled_scores.sort()
            units = np.load(dirname + """/tmp_models/{units}.npy""".format(units=z+"_units"))
            pdf = np.load(dirname + """/tmp_models/{dist}.npy""".format(dist=z))
            plt.title("""Results for '{z}' distribution""".format(z=z))
            plt.plot(np.arange(0,len(scaled_scores)), scaled_scores, color='green')
            plt.plot(np.arange(0,len(units)), units[idx], color='blue')
            plt.plot(np.arange(0,len(pdf)), pdf[idx], color='red')
            plt.show()
    else:

        if os.path.isfile(dirname + """/tmp_models/{score}.npy""".format(score="episode_scores")):
            smoothing_window=20
            episode_scores = np.load(dirname + """/tmp_models/{score}.npy""".format(score="episode_scores"))
            scores_smoothed = pd.Series(episode_scores).rolling(smoothing_window, min_periods=smoothing_window).mean()
            plt.title("""Smoothed episode scores""")
            plt.plot(np.arange(0,len(scores_smoothed)), scores_smoothed, color='green')
            plt.show()

        for z in ('beta', 'cauchy', 'gamma', 'rayleigh', 'weibull'):

            print("""\nPlotting results for '{z}' distribution\n""".format(z=z))

            if os.path.isfile(dirname + """/tmp_models/{score}.npy""".format(score=z+"_scores")):
                scaled_scores = np.load(dirname + """/tmp_models/{score}.npy""".format(score=z+"_scores"))
                idx = scaled_scores.argsort()
                scaled_scores.sort()
                units = np.load(dirname + """/tmp_models/{units}.npy""".format(units=z+"_units"))
                pdf = np.load(dirname + """/tmp_models/{dist}.npy""".format(dist=z))
                plt.title("""Results for '{z}' distribution""".format(z=z))
                plt.plot(np.arange(0,len(scaled_scores)), scaled_scores, color='green')
                plt.plot(np.arange(0,len(units)), units[idx], color='blue')
                plt.plot(np.arange(0,len(pdf)), pdf[idx], color='red')
                plt.show()
