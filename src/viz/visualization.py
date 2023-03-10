import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def score_visualisation(score, title=None, show_variance=False, figure=True):
    if figure:
        plt.figure(figsize=(16, 9))
    if type(score) == pd.core.frame.DataFrame:
        if title is not None:
            plt.title(title)
        plt.ylabel("Scores", fontsize=12)
        plt.xlabel("Training Epochs", fontsize=12)
        plt.yscale('log')
        plt.grid(True)
        for i, c in enumerate(score.columns) :
            avg_score = score.loc[:,c].rolling(50).mean()
            min_score = score.loc[:,c].rolling(50).min()
            max_score = score.loc[:,c].rolling(50).max()
            plt.plot(avg_score, linewidth=3, label = c)
            if show_variance:
                plt.fill_between(np.arange(len(min_score)), list(min_score.to_numpy()),
                                list(max_score.to_numpy()), alpha=0.2)
        plt.legend(loc='best')
    else :
        score = score.reshape(-1, 1)
        for s in range(score.shape[1]):
            if type(score[:, s]) != pd.core.frame.DataFrame:
                s_ = pd.DataFrame(score[:, s])

            avg_score = s_.rolling(50).mean()
            min_score = s_.rolling(50).min()
            max_score = s_.rolling(50).max()
            if title is not None:
                plt.title(title)
            plt.ylabel("Scores", fontsize=12)
            plt.xlabel("Training Epochs", fontsize=12)
            plt.plot(avg_score, color='blue', linewidth=3)
            plt.fill_between(np.arange(len(min_score)), list(min_score.iloc[:, 0]),
                            list(max_score.iloc[:, 0]), alpha=0.2)
            plt.yscale('log')
            plt.grid(True)
