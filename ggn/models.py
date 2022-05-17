import torch
from .distances import diagonal_KL, lowrank_KL
from . import PD_THR

class SphericalEmbedder(torch.nn.Module):
    def __init__(self, n_in, dim=2):
        super(SphericalEmbedder, self).__init__()

        self.n_in = n_in
        self.dim = dim # dimension of the latent space

        # initializations
        mean = torch.randn(self.n_in, self.dim)
        lambdas = torch.ones(self.n_in)

        # trainable parameters
        self.mean = torch.nn.Parameter(mean)
        self.lambdas = torch.nn.Parameter(lambdas)

    def forward(self, x):
        diag = torch.clamp(self.lambdas, PD_THR, torch.inf) # ensure PD
        diag = torch.tile(diag.unsqueeze(-1), (1, self.dim))

        energies = diagonal_KL(x, self.mean, diag)
        energies = energies.T # <<< need for inverse KL

        return energies

class DiagonalEmbedder(torch.nn.Module):
    def __init__(self, n_in, dim=2):
        super(DiagonalEmbedder, self).__init__()

        self.n_in = n_in
        self.dim = dim # dimension of the latent space

        # initializations
        mean = torch.randn(self.n_in, self.dim)
        diag = torch.ones(self.n_in, self.dim)

        # trainable parameters
        self.mean = torch.nn.Parameter(mean)
        self.diag = torch.nn.Parameter(diag)

    def forward(self, x):
        diag = torch.clamp(self.diag, PD_THR, torch.inf) # ensure PD

        energies = diagonal_KL(x, self.mean, diag)
        energies = energies.T # <<< need for inverse KL

        return energies

class LowRankEmbedder(torch.nn.Module):
    def __init__(self, n_terms, rank=2, dim=2):
        super(LowRankEmbedder, self).__init__()

        self.n_terms = n_terms
        self.dim = dim # dimension of the latent space
        self.rank = rank

        mean = torch.randn(n_terms, self.dim)
        diag = torch.ones(n_terms, self.dim)
        covm = torch.randn(n_terms, self.dim, self.rank)

        self.mean = torch.nn.Parameter(mean)
        self.diag = torch.nn.Parameter(diag)
        self.covm = torch.nn.Parameter(covm)

    def forward(self, x):
        # clamp diagonals for ensuring PSD matrices
        diag = torch.clamp(self.diag, 0.01, torch.inf)

        energies = lowrank_KL(x, self.mean, diag, self.covm)
        energies = energies.T # <<< only needed for inverse KL

        return energies
