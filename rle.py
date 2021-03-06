"""run length encoding for kaggle
"""

from __future__ import division, print_function
import itertools
import numpy as np # linear algebra

import pdb

# Fast run length encoding
def rlencode(x):
    """https://pythonexample.com/code/numpy-run-length-encoding/
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the R rle function.
     
    Parameters
    ----------
    x : 1D array_like
        Input array to encode
     
    Returns
    -------
    start positions, run lengths, run values
     
    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int), 
                np.array([], dtype=int), 
                np.array([], dtype=x.dtype))
 
    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
     
    return starts, lengths, values

def myrlestring(x):
	starts, lengths, values = rlencode(x)
	starts = np.array(starts) + 1
	out = ''
	for idx, val in enumerate(values):
		if val == 1:
			out += ' %i %i' % (starts[idx], lengths[idx])
	return out[1:]

def test_myrlestring():
	right = np.concatenate([np.ones((100,100)), np.zeros((100,100))])
	left = np.concatenate([np.zeros((100,100)), np.ones((100,100))])
	whole = np.concatenate([left, right],axis=1)
	encoding = myrlestring(np.reshape(whole.transpose(), np.prod(whole.shape)))
	pdb.set_trace()


if __name__ == '__main__':
	test_myrlestring()