__version__ = '0.1.0'

import os

# minimum value allow in diagonal for ensuring positive definitiveness
PD_THR = 0.01

# toygraph data, used only for usage examples
basename       = os.path.dirname(__file__)
TOYGRAPH       = os.path.join(basename, 'toy.npz')
TOYGRAPH_means = os.path.join(basename, 'TOYGRAPH_means.npy')
TOYGRAPH_diags = os.path.join(basename, 'TOYGRAPH_diags.npy')
TOYGRAPH_covms = os.path.join(basename, 'TOYGRAPH_covms.npy')
