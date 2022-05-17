import sys

import torch
import numpy as np
import scipy

from . import PD_THR
from .models import SphericalEmbedder, DiagonalEmbedder, LowRankEmbedder
from .utils import MyCollator, fit

def ggn():
    if len(sys.argv) < 8:
        print('Arguments missing.')
        print('Usage . [sp] [dim] [k] [nepochs] [batch_sz] [cuda] [prefix]')
        return sys.exit(1)

    sp_fin   = sys.argv[1]
    dim      = int(sys.argv[2])
    k        = int(sys.argv[3])
    nepochs  = int(sys.argv[4])
    batch_sz = int(sys.argv[5])
    cuda     = bool(sys.argv[6]) and torch.cuda.is_available()
    prefix   = sys.argv[7]
    saveall  = -1

    if len(sys.argv) > 8:
        saveall  = 1

    # read sparse matrix with the shortest path lengths
    spmat   = scipy.sparse.load_npz(sp_fin)
    n_nodes = spmat.shape[0]

    if k  < 0:
        embedder = SphericalEmbedder(n_nodes, dim=dim)
    elif k == 0:
        embedder = DiagonalEmbedder(n_nodes, dim=dim)
    else:
        embedder = LowRankEmbedder(n_nodes, dim=dim, rank=k)

    if cuda:
        embedder = embedder.cuda()

    # create dataloader
    ds = torch.utils.data.DataLoader(
        [torch.LongTensor([i]) for i in range(n_nodes)],
        batch_size=batch_sz,
        shuffle=True,
        drop_last=False,
        collate_fn=MyCollator(spmat, cuda)
    )

    if saveall == -1:
        losses, times = fit(embedder, ds, epochs=nepochs, verbose=True)
    else:
        # save the weights learned at each epoch
        losses, times = fit(embedder, ds, epochs=nepochs, verbose=True, prefix=prefix)

    # save loss & learned weights
    np.save(f'{prefix}loss.npy', np.array(losses))

    for n, p in embedder.named_parameters():
        p = p.detach().cpu().numpy() # move to cpu

        if n == 'lambdas':
            p = np.tile(p[:,np.newaxis], (1, dim))
            n = 'diag'

        if n == 'diag':
            p = np.clip(p, PD_THR, np.inf)

        np.save(f'{prefix}{n}.npy', p)
