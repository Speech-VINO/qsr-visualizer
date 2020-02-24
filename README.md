# Power BI App for Visualizing Dataset behaviour

In this project, a power bi based visualization app is proposed to analyze the neighborhood of the dataset using Quality Spatial Relations (QSR). Two datasets have been used to be analysed under Monte Carlo Tree Search (MCTS) and also other popular Machine Learning algorithms. MCTS uses a UCB (Upper Confidence Bound) based reward system that simulates within the state space of every dataset. 

## Datasets used

From Speech project, [The RAVDESS dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) based on the paper [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391) has been analysed using [speech_sigproc.py](./speech/audio/speech_sigproc.py) 

## State Space Model

A state space model consisting of features and labels are fed into an MCTS algorithm. The state space model for each dataset consists of observations and transitions. When the data transitions to another state from an initial random state, the UCM model which is a multi-armed action bandit model, executes a Monte Carlo learning for `n` simulations. After obtaining the final state from an input state, the final state looks at the defined clusters which are ordered by a chosen metric. The number of maximum clusters are defined in the app user interface. The maximum clusters should not exceed the state space dimension of the input data.

## Data Processing

For a dataset, the feature extraction is the first step. An Expectation Maximisation step has likelihoods in various forms. For factor analysis, there is a marginal likelihood; for dirichlet processes, the likelihood is a differential equation. Since the EM step is Gaussian in nature, the probability distribution(s) is/are used for clustering and searching the dataset. 

## Data Wrangling

In the data wrangling stage, the data is processed through a complex plane where there is timestamp information recorded from the output of UCB scores from a Monte Carlo Simulation. Also, the time series is unit spaced with each recorded variable being: 

- no. of units required for calibration, 
- the chosen distribution,
- scores that are scaled from 0 to 1



## Multi-armed bandit frameworks

In a multi-armed bandit framework, a person simulates the environment by taking actions from a multi-labeled data source. The reward tree structure considers two approaches that have been documented based on:

- Least Noise model, in the case of Factor Analysis
- Maximum Similarity model, in the case of Latent Semantic Analysis

![Dataset Visualizer](./screenshot.PNG)

![Graph Statistics](./barplot.PNG)