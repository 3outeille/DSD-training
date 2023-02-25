import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np


class DSDTraining(nn.Module):
    def __init__(self, model, sparsity, train_on_sparse=False):
        super(DSDTraining, self).__init__()

        self.model = model
        self.sparsity = sparsity
        self.train_on_sparse = train_on_sparse

        # Get only fc layers.
        tmp = list(self.model.named_parameters())
        self.layers = []
        for i in range(2, len(tmp), 2):
            w, b = tmp[i], tmp[i + 1]
            if "fc" in w[0] or "fc" in b[0]:
                self.layers.append((w[1], b[1]))

        # Init masks
        self.reset_masks()

    def reset_masks(self):
        self.masks = []
        for w, b in self.layers:
            mask_w = torch.ones_like(w, dtype=bool)
            mask_b = torch.ones_like(b, dtype=bool)
            self.masks.append((mask_w, mask_b))

        return self.masks

    def update_masks(self):
        for i, (w, b) in enumerate(self.layers):
            q_w = np.quantile(torch.abs(w).detach().numpy(), q=self.sparsity)
            mask_w = torch.where(torch.abs(w) < q_w, True, False)

            q_b = np.quantile(torch.abs(b).detach().numpy(), q=self.sparsity)
            mask_b = torch.where(torch.abs(b) < q_b, True, False)

            self.masks[i] = (mask_w, mask_b)

    def forward(self, x):
        return self.model(x)
