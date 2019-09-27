# -*- coding: utf-8 -*-
# This file is a personal adaptation the SPORCO package. Details of the
# copyright for the toolbox and user license can be found in the 'LICENSE.txt'
# file distributed with the package.

"""Classes for ADMM algorithm for the Convolutional BPDN problem specific to the"
    Object project. Adaptation from SPORCO."""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from builtins import range

import copy
from types import MethodType

import numpy as np
import tensorly as tl
from scipy.linalg import block_diag

from sporco.admm import admm
from sporco.admm import cbpdn
import sporco.cnvrep as cr
import sporco.linalg as sl
import sporco.prox as sp
from sporco.util import u


__author__ = """David Reixach <dreixach@iri.upc.edu>"""

class AKConvBPDN(object):
    """Boundary masking for convolutional representations using the
    Alternated Kruscal ConvBPDN technique described in
    :cite:`humbert-2019`. Implemented as a wrapper about a
    ConvBPDN or derived object (or any other object with
    sufficiently similar interface and internals). The wrapper is largely
    transparent, but must be taken into account when setting some of the
    options for the inner object, e.g. the shape of the ``L1Weight``
    option array must take into account the extra dictionary atom appended
    by the wrapper.
    """

    def __init__(self, D0, S, R, xmethod='convbpdn', lmbda=None, optx=None,
                dimK=None, dimN=2,*args, **kwargs):
        """
        Parameters
        ----------
        xstep : internal xstep object (e.g. xstep.ConvBPDN)
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        R : array_like
          Rank array
        lmbda : list of float
          Regularisation parameter
        opt : list containing :class:`ConvBPDN.Options` object
          Algorithm options for each individual solver
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        *args
          Variable length list of arguments for constructor of internal
          xstep object (e.g. mu)
        **kwargs
          Keyword arguments for constructor of internal xstep object
        """

        # Parse mu
        if 'mu' in kwargs:
            mu = kwargs['mu']
        else:
            mu = None

        # Number of channel dimensions
        if self.xmethod.lower() != 'convbpdn' and self.xmethod.lower() !='convelasticnet':
            raise ValueError('Parameter xmethod accepted values are: ''ConvBPDN'' '
                                'and ''ConvElasticNet''')

        # Infer outer problem dimensions
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Decomposed Kruskal Initialization
        self.R = R
        self.Z = list()
        for i,Nvi in enumerate(self.cri.Nv):                    # Ui
            self.Z.append(np.random.randn(Nvi,np.sum(self.R)))

        # Store initial Dictionary
        self.setdict(D0)

        # Store input signal
        self.S = S

        # Construct inner xstep object
        self.xstep = list()
        for l in range(self.cri.dimN)
            D0c = self.convolvedict(l)
            Sl = self.reshapesignal(l)
            if self.xmethod.lower() != 'convbpdn':
                xstep.append(cbpdn.ConvBPDN(D0c, Sl, lmbda[l], optx[l],
                    dimK=self.cri.dimK, dimN=1))
            else:
                xstep.append(cbpdn.ConvElasticNet(D0c, Sl, lmbda[l], mu[l],
                    optx[l], dimK=self.cri.dimK, dimN=1))

        # Required because dictlrn.DictLearn assumes that all valid
        # xstep objects have an IterationStats attribute
        self.IterationStats = self.xstep.IterationStats

        # # Record ystep method of inner xstep object
        # self.inner_ystep = self.xstep.ystep
        # # Replace ystep method of inner xstep object with outer ystep
        # self.xstep.ystep = MethodType(AKConvBPDN.ystep, self)
        #
        # # Record obfn_gvar method of inner xstep object
        # self.inner_obfn_gvar = self.xstep.obfn_gvar
        # # Replace obfn_gvar method of inner xstep object with outer obfn_gvar
        # self.xstep.obfn_gvar = MethodType(AKConvBPDN.obfn_gvar, self)


    def solve(self):
        """Call the solve method of the inner cbpdn object and return the
        result.
        """

        # Call solve method of inner xstep object
        Xi = self.xstep.solve()

        # Copy attributes from inner xstep object
        self.timer = self.xstep.timer
        self.itstat = self.xstep.itstat

        # Return result of inner xstep object
        return Xi


    def setdict(self, D):
    """Set dictionary array."""

        self.D = np.asarray(D)


    def expandZRankDec(self,l=None):
    """Expand Kruskal Zl (Rank Decomposed)."""

        Nv = self.cri.Nv
        R = self.R

        if l is not None:
            Nv = np.delete(Nv,[l-1])            # remove l dimension
            R = np.delete(R,[l-1])

        Nv.append(np.sum(Rv))

        Tz = tl.tenalg.khatri_rao(self.Z)

        return tl.base.vec_to_tensor(Tz,Nv)     # as array Tz(N0,N1,...,K*R)


    def expandZ(self):
    """Expand Kruskal Z."""

        Nv = self.cri.Nv
        R = self.R

        Nv.append(np.sum(Rv))

        Tzk = np.dot(tl.tenalg.khatri_rao(self.Z),self.getweights)

        return tl.base.vec_to_tensor(Tzk,Nv)     # as array Tz(N0,N1,...,K)


    def convolvedict(self,l=None):
    """~D: Convolve D w/ Zl."""

        axisN = self.cri.axisN
        N = self.cri.N
        Nv = self.cri.Nv
        M = self.cri.M                              # Number of filters K
        dsz = self.cri.dsz                          # Diccionary Size

        Tz = self.expandZRankDec(l)

        # TzFull
        Tzlst = list()
        for i in range(dsz[l-1]):
            Tzlst.append(Tz)                        # Z for each j

        TzFull = np.stack(Tzlst,axis=l-1)           # As array Tz(N0,N1,...,Nl,...,K*R)

        # DzFull
        DFull = np.reshape(self.D,[N,M],order='F')
        DFull = np.dot(DFull,np.transpose(self.getweights))
        DFull = np.reshape(DFull,[Nv,np.sum(self.R)],order='F') # As array D(N0,N1,...,K*R)

        # Purge l
        if l is not None:
            axisN = np.delete(axisN,[l-1])       # remove l dimension
            Nv = np.delete(Nv,[l-1])

        # Convolve
        Xf = sl.rfftn(TzFull,None,axisN)
        Df = sl.rfftn(DFull,Nv,axisN)
        Dcf = np.multiply(Df,Xf)
        Dc = sl.irrftn(Dcf,Nv,axisN)

        Dc = np.moveaxis(Dc,l-1,0)              # spatial dimension at first place

        return np.reshape(Dc,[dsz[l-1],np.prod(Nv),np.sum(self.R)],order='F')) # As array Dc(Dl,N',K*R)


    def getweights(self):
        """Linear map from [NxR*K] to [NxK] array."""

        weightsArray = list()
        for k,Rk in enumerate(self.R):
            weightsArray.append(np.ones([Rk,1]))

        return block_diag(*weightsArray)                 # map from R*K to K


    def reshapesignal(self,l):
        """Reshape S from [N1xN2x...xNp] to [NlxC] array."""

        Nv = self.cri.Nv
        C = int(self.cri.N/Nv[l-1])
        Q = self.cri.K              # multi-signal

        Sl = np.moveaxis(self.S,l-1,0)

        return np.reshape(Sl,[Nv[l-1],C,Q],order='F').squeeze()  # As array S(Nl,C,Q)


    def getcoef(self):
        """Get result of inner xstep object."""

        return self.expandZ()


    def getKruskal(self):
        """Get decomposed Krukal Z."""

        return self.Z()


    def reconstruct(self, X=None):
        """Reconstruct representation."""

        Tz = self.expandZ()

        Xf = sl.rfftn(Tz,None,self.cri.axisN)
        Df = sl.rfftn(self.D,Nv,self.cri.axisN)     # Self.Df is not stored
        Sf = np.sum(Df*Xf, axis=self.cri.axisM)

        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)


    def getitstat(self):
        """Get iteration stats from inner xstep object."""

        return self.xstep.getitstat()
