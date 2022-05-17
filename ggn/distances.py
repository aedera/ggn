import torch

def diagonal_KL(idx, mP, DP):
  """
  KL(P|Q)=0.5[log|Dq|/log|Dp| + tr(inv(Dq)@Dp) + delta.T@inv(Dq)@delta - d],

  where delta = mQ - mP and d is the dimensionality.

  Args:
    mP [batch_sz1, dim_sz]
      The means of the distributions P
    DP [batch_sz1, dim_sz]
      The diagonal covariance matrices of the distributions P
    mQ [batch_sz2, dim_sz]
      The means of the batch_sz2 distributions Q
    DQ [batch_sz2, dim_sz]
      The diagonal covariance matrices of the distributions Q

  Returns:
    KLs [batch_sz1, batch_sz2]
      A tensor of KL divergences [KL(pi|qj)] where i in 0..batch_sz1 and j in 0..batch_sz2.
  """

  mQ, DQ = mP[idx,...], DP[idx,...]

  term1 = DQ.log().sum(-1).unsqueeze(-1) - DP.log().sum(-1).unsqueeze(0)
  term2 = (DP.unsqueeze(0) / DQ.unsqueeze(1)).sum(-1)
  delta = (mQ.unsqueeze(1) - mP.unsqueeze(0))
  term3 = (delta * DQ.unsqueeze(1).rsqrt()).pow(2).sum(-1)

  KLs = (0.5 * (term1 + term2 + term3 - mP.shape[-1])).transpose(0, 1)
  return KLs

def _batch_capacitance_tril(W, D):
    """
    Computes Cholesky of `I + W.T @ inv(D) @ W`

    Args:
      W [batch_sz, n, m]
        a batch of matrices.
      D [batch_sz, n]
        a batch of vectors.
    """
    m = W.size(-1)
    Wt_Dinv = W.transpose(-2,-1) / D.unsqueeze(-2)
    K = torch.matmul(Wt_Dinv, W).contiguous()
    K.view(-1, m * m)[:, ::m + 1] += 1  # add identity matrix to K
    return torch.linalg.cholesky(K)

def lowrank_KL(idx, pm, pD, pW):
  """
  Compute the KL divergence between two low-rank Gaussian distributions:

  KL(p|q)=\int p(x)\log\frac{p(x)}{q(x)} dx

  The covariance matrix of each Gaussian distribution is defined as

  pD[i,:].diag() + pW[i,:] @ pW[i,:].transpose(1,0)

  Args:
    idx LongTensor shape=(batch_sz)
      Indexes to obtain the distribution "p".
    Distributions
      qm Tensor shape=(N, d), N: number of distributions
      qD Tensor shape=(N, d)
      qW Tensor shape=(N, d, K)

  Returns:
    Tensor shape=(B, N)
  """
  # Retrieve p distributions
  qm = pm[idx,...]
  qD = pD[idx,...]
  qW = pW[idx,...]

  # Computing the KL involves calculating three major terms
  #
  # KL = 0.5 * (term1 + term2 + term3 - d)
  #
  # term1: \log\frac{\Sigma_q^{-1}}{\Sigma_p}
  # term2: (\mu_q - \mu_p)^T \Sigma_q^{-1} (\mu_q - \mu_p)
  # term3: \log\frac{|\Sigma_q|}{|\Sigma_p|}

  # Calculate the Cholesky factor of the capacitance matrix
  pC_tril = _batch_capacitance_tril(pW, pD) # (N, K, K)
  qC_tril = pC_tril[idx,...] # (B, K, K)

  ########
  # term 1
  term11 = 2*qC_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) # (N,)
  term12 = qD.log().sum(-1) # (N,)
  term13 = 2*pC_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) # (B,)
  term14 = pD.log().sum(-1) # (B,)

  term1 = term11.unsqueeze(0) + term12.unsqueeze(0) - term13.unsqueeze(1) - term14.unsqueeze(1) # (B, N)

  ########
  # term 2
  # pD (B, d) -> (B, 1, d)
  # qD (N, d) -> (1, N, d)
  term21 = pD.unsqueeze(1)/qD.unsqueeze(0) # (B, N, d)
  # trace
  term21 = term21.sum(-1) # (B, N)

  # qD (N, d)    -> (1, N, d, 1)
  # pW (B, d, K) -> (B, 1, d, K)
  term22 = qD.rsqrt().unsqueeze(0).unsqueeze(-1)*pW.unsqueeze(1) # (B, N, d, K)

  # trace
  B, d, K = pW.shape
  N = qD.shape[0]
  term22 = term22.view(B, N, d*K).pow(2).sum(-1) # (B, N)

  # qW (N, d, K) -> (N, K, d)
  # qD (N, d)    -> (N, 1, d)
  qWt_qDinv = qW.transpose(-1,-2)/qD.unsqueeze(-2) # (N, K, d)

  # qC_tril (N, K, K)
  A = torch.triangular_solve(qWt_qDinv, qC_tril, upper=False)[0] # (N, K, d)
  A = A.contiguous()

  # A  (N, K, d) -> (1, N, K, d)
  # pD (B, d)    -> (B, 1, K, 1)
  term23 = A.unsqueeze(0)*pD.sqrt().unsqueeze(1).unsqueeze(2) # (B, N, K, d)

  # trace
  term23 = term23.view(B, N, d*K).pow(2).sum(-1) # (B, N)

  # A  (N, K, d) -> (1, N, K, d)
  # pW (B, d, K) -> (B, 1, K)
  term24 = A.unsqueeze(0).matmul(pW.unsqueeze(1)) # (B, N, K, K)

  # trace
  term24 = term24.view(B, N, K*K).pow(2).sum(-1) # (B, N)

  term2 = term21 + term22 - term23 - term24 # (B, N)

  ########
  # term 3
  delta = qm.unsqueeze(0) - pm.unsqueeze(1) # (B, N, d)
  term31 = (delta.pow(2) / qD.unsqueeze(0)).sum(-1) # (B, N)

  term32 = A.unsqueeze(0).matmul(delta.unsqueeze(-1)) # (B, N, K, 1)
  term32 = term32.squeeze(-1).pow(2).sum(-1) # (B, N)

  term3 = term31 - term32 # (B, N)

  return 0.5 * (term1 + term2 + term3 - pm.shape[1]) # (B, N)
