import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def score_visualisation(score, title=None):
    score = score.reshape(-1, 1)
    for s in range(score.shape[1]):
        if type(score[:, s]) != pd.core.frame.DataFrame:
            s_ = pd.DataFrame(score[:, s])

        avg_score = s_.rolling(50).mean()
        min_score = s_.rolling(50).min()
        max_score = s_.rolling(50).max()
        plt.figure(figsize=(16, 9))
        if title is not None:
            plt.title(title)
        plt.ylabel("Scores", fontsize=12)
        plt.xlabel("Training Epochs", fontsize=12)
        plt.plot(avg_score, color='blue', linewidth=3)
        plt.fill_between(np.arange(len(min_score)), list(min_score.iloc[:, 0]),
                         list(max_score.iloc[:, 0]), alpha=0.2)
        plt.yscale('log')
        plt.grid(True)
        plt.show()
