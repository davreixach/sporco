#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: cbpdndl.ConvBPDNDictLearn (colour images,
multi-scale colour dictionary).
"""


from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np


from sporco.fista import cbpdndl
from sporco import util
from sporco import plot


# Training images
exim = util.ExampleImages(scaled=True, zoom=0.5)
img1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
img2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
img3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S = np.stack((img1, img2, img3), axis=3)


# Highpass filter test images
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(16, 16, 3, 96)


# Set ConvBPDNDictLearn parameters, including multi-scale dictionary size
lmbda = 0.2
L_sc = 360.0
L = 50.0
dsz = ((8,8,3,32), (12,12,3,32), (16,16,3,32))
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
                                         'DictSize': dsz,
                                         'CBPDN': {'BackTrack' : { 'Enabled' : True }, 'L' : L_sc},
                                         'CCMOD': {'ZeroMean': True, 'BackTrack' : { 'Enabled' : True },
                                         'L' : L } })


# Run optimisation
d = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt)
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'), "\n")


# Display dictionaries
D1 = D1.squeeze()
fig1 = plot.figure(1, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D1, dsz), fgrf=fig1, title='D1')
fig1.show()


# Plot functional value and residuals
its = d.getitstat()
fig2 = plot.figure(2, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plot.subplot(1,3,2)
plot.plot(np.vstack((its.X_Rsdl, its.D_Rsdl)).T,
          fgrf=fig2, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['X', 'D'])
plot.subplot(1,3,3)
plot.plot(np.vstack((its.X_L, its.D_L)).T, fgrf=fig2, xlbl='Iterations',
          ylbl='Inverse of Gradient Step Parameter', ptyp='semilogy',
          lgnd=['$\L_X$', '$\L_D$'])
fig2.show()


# Wait for enter on keyboard
input()