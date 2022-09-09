import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap(mat,title = None):
        
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    im = ax.imshow(mat, cmap='coolwarm', interpolation='nearest')

    ax.set_xticks(np.arange(mat.shape[1]), labels = list(mat.columns))
    ax.set_yticks(np.arange(mat.shape[0]), labels = list(mat.index))


    # Loop over data dimensions and create text annotations.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = ax.text(j, i, mat.iloc[i, j],
                           ha="center", va="center", color="w",
                           size = 10
                          )


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)

    plt.show()