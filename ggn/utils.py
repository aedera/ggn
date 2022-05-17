import scipy.sparse
import numpy as np
import torch
import time
from . import PD_THR

EPS = 1e-7

class MyCollator(object):
   def __init__(self, splengths, cuda):
      self.splengths = splengths # sparse matrix containing shortest path lengths

      # remove inf elements if any
      self.splengths[self.splengths == np.inf] = 0.0

      # mark elements on the main diagonal to be excluded from the loss function
      self.splengths.setdiag(torch.nan)
      # recalculate sparse structure
      self.splengths = scipy.sparse.csr_matrix(self.splengths)
      self.cuda = cuda

   def __call__(self, term_ids):
       """
       Creates a batch on the fly from a given set of term IDs. For each input term,
       a label vector is defined as the shortest paths between it and each of the remaining terms.

       Args:
         term_ids torch.Tensor[torch.Long]
           The term_ids is a tensor of long integer values, each representing a term id.

       Returns:
         X, Y list(torch.Tensor, torch.Tensor)
           X contains term ids whereas Y contains its corresponding shortest paths
       """
       X = torch.stack(term_ids).flatten()

       # retrieve shortest paths calculated for each x in X
       idx = X.cpu()
       anc = torch.from_numpy(self.splengths[idx,:].todense()).float()
       des = torch.from_numpy(self.splengths.T[idx,:].todense()).float()

       if self.cuda:
          X   = X.to('cuda:0')
          anc = anc.to('cuda:0')
          des = des.to('cuda:0')

       Y = anc - des # indicate descendents with negative hops

       return X, Y

def loss_fn(energies, y_true, beta=1):
    pos  = torch.square(energies) * y_true.eq(1)
    neg  = torch.exp(-beta*energies) * y_true.eq(-1) # unreachable nodes
    neg += torch.exp(-beta*energies) * y_true.eq(0) # unreachable nodes

    loss = pos.sum(1) + neg.sum(1)
    loss = loss.mean()

    y_true[y_true.isnan()] = 0 # mask nan values for calculating min/max operations
    batch_idxs = torch.arange(y_true.shape[0], device=y_true.device)

    # second (upward ranking preservation)
    max_hops, idxs = y_true.max(1, keepdim=True)
    rank_true = torch.divide(y_true * y_true.gt(0), max_hops+EPS)

    max_vals = energies[batch_idxs,
                        idxs.squeeze()].unsqueeze(-1)
    rank_pred = torch.divide(energies * y_true.gt(0), max_vals+EPS)
    loss += torch.square(rank_true - rank_pred).sum(1).mean()

    # third (downward ranking preservation)
    min_hops, idxs = y_true.min(1, keepdim=True)
    rank_true = torch.divide(y_true * y_true.lt(0), min_hops+EPS)

    min_vals = energies[batch_idxs,
                        idxs.squeeze()].unsqueeze(-1)
    rank_pred = torch.divide(energies * y_true.lt(0), min_vals+EPS)
    loss += torch.square(rank_true - rank_pred).sum(1).mean()

    return loss


def fit(model, dataset, lr=0.001, epochs=500, verbose=True, prefix=None):
  model.train()

  params = model.parameters()
  optm   = torch.optim.Adam(params, lr=lr)

  losses = []
  times = []
  for epoch in range(epochs):
    loss_ = []

    start_time = time.time()
    for x, y_true in dataset:
      y_pred = model(x)
      loss = loss_fn(y_pred, y_true, beta=1)

      loss.backward()
      optm.step()
      optm.zero_grad()

      loss_.append(loss.item())

    if prefix is not None:
       for n, p in model.named_parameters():
          p = p.detach().cpu().numpy() # move to cpu

          if n == 'diag':
             p = np.clip(p, PD_THR, np.inf)

          np.save(f'{epoch:010d}_{prefix}{n}.npy', p)

    #scheduler.step()
    losses.append(np.mean(loss_))
    times.append(time.time() - start_time)

    if verbose and epoch % 5 == 0:
       print("---Epoch %d: Loss %.4f %.2f sec ---" % (epoch, losses[-1], times[-1]))

  return losses, times
