import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def plot_wb(model, fig_path, ranges=None):

    tmp = list(model.named_parameters())
    layers = []
    for i in range(0, len(tmp), 2):
          w, b = tmp[i], tmp[i + 1]
          if ("conv" in w[0] or "conv" in b[0]) or ("fc" in w[0] or "fc" in b[0]):
            layers.append((w, b))

    num_rows = len(layers)

    fig = plt.figure(figsize=(20, 40))

    i = 1
    for w, b in layers:
        w_flatten = w[1].flatten().detach().cpu().numpy()
        b_flatten = b[1].flatten().detach().cpu().numpy()

        fig.add_subplot(num_rows, 2, i)
        plt.title(w[0])
        plt.hist(w_flatten, bins=100, range=ranges);

        fig.add_subplot(num_rows, 2, i + 1)
        plt.title(b[0])
        plt.hist(b_flatten, bins=100, range=ranges);

        i += 2
    
    fig.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def set_all_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)