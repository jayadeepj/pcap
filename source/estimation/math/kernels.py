import torch
import numpy as np


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        # lower bandwidth increases the uncertainty, higher b/w means the edges become sharp, i.e. lower uncertainty
        self.sigma = sigma
        print(f"Using RBF Kernel of sigma = {self.sigma} ")

    def forward(self, x, y):
        xx = x.matmul(x.t())
        xy = x.matmul(y.t())
        yy = y.matmul(y.t())

        # the dnorm2 is the squared euclidean distance in larger ranks
        # this follows the x^2 + y^2 -2xy format
        # notice that here dnorm2 is a tensor rather than a single value
        dnorm2 = -2 * xy + xx.diag().unsqueeze(1) + yy.diag().unsqueeze(0)

        # apply the median heuristic (pytorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            med_val = np.median(np_dnorm2)
            h = med_val / (2 * np.log(x.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        gd = -gamma * dnorm2

        k_xy = gd.exp()

        return k_xy
