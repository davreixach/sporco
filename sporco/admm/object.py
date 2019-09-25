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

from sporco.admm import admm
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

    def __init__(self, xstep, D, S, R, *args, **kwargs):
        """
        Parameters
        ----------
        xstep : internal xstep object (e.g. xstep.ConvBPDN)
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        R : list_like
          Rank list
        # W : array_like
        #   Mask array. The array shape must be such that the array is
        #   compatible for multiplication with input array S (see
        #   :func:`.cnvrep.mskWshape` for more details).
        *args
          Variable length list of arguments for constructor of internal
          xstep object
        **kwargs
          Keyword arguments for constructor of internal xstep object
        """

        # Number of channel dimensions
        if 'dimK' in kwargs:
            dimK = kwargs['dimK']
        else:
            dimK = None

        # Number of spatial dimensions
        if 'dimN' in kwargs:
            dimN = kwargs['dimN']
        else:
            dimN = 2

        # Infer problem dimensions
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # # Construct impulse filter (or filters for the multi-channel
        # # case) and append to dictionary
        # if self.cri.Cd == 1:
        #     self.imp = np.zeros(D.shape[0:dimN] + (1,))
        #     self.imp[(0,)*dimN] = 1.0
        # else:
        #     self.imp = np.zeros(D.shape[0:dimN] + (self.cri.Cd,)*2)
        #     for c in range(0, self.cri.Cd):
        #         self.imp[(0,)*dimN + (c, c,)] = 1.0
        # Di = np.concatenate((D, self.imp), axis=D.ndim-1)

        # Construct inner xstep object
        self.xstep = xstep

        # Required because dictlrn.DictLearn assumes that all valid
        # xstep objects have an IterationStats attribute
        self.IterationStats = self.xstep.IterationStats

        # Kruskal Decomposition Parameters
        self.R = R
        self.uniformRank = True
        self.Z = list()

        for k,Rk in enumerate(self.R,1):            # for each k
            self.uniformRank = self.uniformRank and Rk == self.R[k]
            Uk = list()
            for i,Nvi in enumerate(self.cri.Nv):    # Ui
                Uk.append(np.random.randn(Nvi,Rk)

            self.Z.append(Uk)

        # # Mask matrix
        # self.W = np.asarray(W.reshape(cr.mskWshape(W, self.cri)),
        #                     dtype=self.xstep.dtype)
        # # If Cd > 1 (i.e. a multi-channel dictionary) and mask has a
        # # non-singleton channel dimension, swap that axis onto the
        # # dictionary filter index dimension (where the
        # # multiple-channel impulse filters are located)
        # if self.cri.Cd > 1 and self.W.shape[self.cri.dimN] > 1:
        #     self.W = np.swapaxes(self.W, self.cri.axisC, self.cri.axisM)

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



    # def setdict(self, D=None):
    #     """Set dictionary array."""
    #
    #     # Di = np.concatenate((D, sl.atleast_nd(D.ndim, self.imp)),
    #     #                     axis=D.ndim-1)
    #     self.xstep.setdict(D)

    def setdict(self, D=None):
    """Set dictionary array."""

    if D is not None:
        self.D = np.asarray(D, dtype=self.dtype)
    # self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
    # # Compute D^H S
    # self.DSf = np.conj(self.Df) * self.Sf
    # if self.cri.Cd > 1:
    #     self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
    # if self.opt['HighMemSolve'] and self.cri.Cd == 1:
    #     self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), self.rho,
    #                               self.cri.axisM)
    else:
        self.c = None

    def expandZRankDec(self,l=None):

        Nv = self.cri.Nv
        if l is not None: del Nv[l-1]       # remove l dimension

        Tzlst = list()
        for k,Uk in enumerate(self.Z):      # for each k
            if l is not None: del Uk[l-1]   # remove l dimension

            Tzk = tl.tenalg.khatri_rao(Uk)
            Nvk = Nv
            Nvk.append(self.R[k])           # rank-decomposed
            Tzlst.append(tl.base.vec_to_tensor(Tzk,Nvk))

        if self.uniformRank:
            return np.stack(Tzlst,axis=self.cri.dimN+1)  # as array Tz(N0,N1,...,K,R)
        else:
            return Tzlst                                 # as list Tzk(N0,N1,...,R)


    def expandZ(self):

        Nv = self.cri.Nv

        Tzlst = list()
        for k,Uk in enumerate(self.Z):      # for each k
            weights = np.ones([self.R[k],1])
            if l is not None: del Uk[l-1]   # remove l dimension

            Tzk = np.dot(tl.tenalg.khatri_rao(Uk),weights)
            Tzlst.append(tl.base.vec_to_tensor(Tzk,Nv))

        return np.stack(Tzlst,axis=self.cri.dimN)  # as array Tz(N0,N1,...,K)



    def convolvedict(self,l=None):

        axisN = self.cri.axisN
        Nv = self.cri.Nv

        Tz = self.expandZRankDec(l)

        # TzFull and DFull
        Tzlst = list()
        Dlst = list()
        if self.uniformRank:
            for i in range(Nv[l-1]):
                Tzlst.append(Tz)                   # Z for each j

            TzFull = np.stack(Tzlst,axis=l-1)           # As array

            for r in range(self.R[0]):
                Dlst.append(self.D)

            DFull = np.stack(Dlst,axis=self.cri.dimN+1) # As array  D(N0,N1,...,K,R)

        else:
            TzFull = list()
            DFull = list()
            for k,Tzk in enumerate(Tz):            # for each k
                for i in range(Nv[l-1]):
                    Tzlst.append(Tzk)              # Z for each j

                TzFull.append(np.stack(Tzlst,axis=l-1)) # As list

            for k,Rk in enumerate(self.R):              # for each k
                for r in range(Rk):
                    Dlst.append(np.take(self.D,k,self.cri.dimN))

                DFull.append(np.stack(Dlst,axis=self.cri.dimN+1))   # As list Dk(N0,N1,...,R)

        # Purge l
        if l is not None:
            del axisN[l-1]                              # remove l dimension
            del Nv[l-1]

        # Convolve
        if self.uniformRank:
            Xf = sl.rfftn(TzFull,None,axisN)
            Df = sl.rfftn(self.D,Nv,axisN)
            Dcf = np.multiply(Df,Xf)

        else:
            Dcf = list()
            for k,Tzk in enumerate(Tzfull):              # for each k
                Xfk = sl.rfftn(Tzk,None,axisN)
                Dfk = sl.rfftn(self.D,Nv,axisN)
                Dcf.append(np.multiply(Df,Xf))

        return Dcf


    def getcoef(self):
        """Get result of inner xstep object."""

        return self.expandZ()


    def reconstruct(self, X=None):
        """Reconstruct representation."""

        Tz = self.expandZ(l)

        Xf = sl.rfftn(Tz,None,self.cri.axisN)
        Df = sl.rfftn(self.D,Nv,self.cri.axisN)     # Self.Df is not stored
        Sf = np.sum(Df*Xf, axis=self.cri.axisM)

        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)



    def getitstat(self):
        """Get iteration stats from inner xstep object."""

        return self.xstep.getitstat()
