import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()


def plot_spatio_temporal_data(x, figsize=(20, 8), add_text=False, **textkw):
    """x is a (T, n, n) tensor, T is the temporal dimension, nxn is the spatio dimension"""
    T = x.shape[0]
    n = x.shape[1]

    # each row can at most have 5 images
    if T % 5 == 0:
        fig, axs = plt.subplots(T // 5, 5, figsize=figsize)
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots((T // 5 + 1), 5, figsize=figsize)
        axs = axs.ravel()


    # row_label = np.arange(n)
    # col_label = np.arange(n)

    # for i in range(T):
    #     im, cbar = heatmap(x[i, ...], row_label, col_label, ax=axs[i], cmap='YlGn')
    #     if add_text:
    #         texts = annotate_heatmap(im, valfmt="{x:.1f}", **textkw)
    #
    # fig.tight_layout()
    # return fig

    for i in range(T):
        sns.heatmap(x[i, ...], ax=axs[i], cbar=False)

    plt.show()
