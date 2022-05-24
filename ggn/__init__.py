__version__ = '0.1.0'

import os

# minimum value allow in diagonal for ensuring positive definitiveness
PD_THR = 0.01

basename = os.path.dirname(__file__)
TOYGRAPH = os.path.join(basename, 'toy.npz')
