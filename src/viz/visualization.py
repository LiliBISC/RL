# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:16:31 2023

@author: lilian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def score_visualisation(score, title):
    score = score.reshape(-1, 1)
    for s in range(score.shape[1]):
        if type(score[:, s]) != pd.core.frame.DataFrame:
            s_ = pd.DataFrame(score[:, s])

        avg_score = s_.rolling(50).mean()
        std_score = s_.rolling(50).std()
        plt.figure(figsize=(16, 9))
        plt.title(title)
        plt.ylabel("Scores", fontsize=12)
        plt.xlabel("Training Epochs", fontsize=12)
        plt.plot(avg_score, color='blue', linewidth=3)
        plt.fill_between(np.arange(len(std_score)), list(std_score.iloc[:, 0]),
                         list(avg_score.iloc[:, 0] + std_score.iloc[:, 0]), alpha=0.2)
        plt.grid(True)
