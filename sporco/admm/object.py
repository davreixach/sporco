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
from builtins import object

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

class IterStatsConfig(object):
    """Configuration object for Alternated Kruscal ConvBPDN learning algorithm
    iteration statistics.

    Adaptation from Sporco ditclearn::IterStatsConfig

    """

    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""


    def __init__(self, isfld, isxmap, evlmap, hdrtxt, hdrmap,
                 fmtmap=None):
        """
        Parameters
        ----------
        isfld : list
          List of field names for iteration statistics namedtuple
        isxmap : dict
          Dictionary mapping iteration statistics namedtuple field names
          to field names in corresponding X step object iteration
          statistics namedtuple
        evlmap : dict
          Dictionary mapping iteration statistics namedtuple field names
          to labels in the dict returned by :meth:`DictLearn.evaluate`
        hdrtxt : list
          List of column header titles for verbose iteration statistics
          display
        hdrmap : dict
          Dictionary mapping column header titles to IterationStats entries
        fmtmap : dict, optional (default None)
          A dict providing a mapping from field header strings to print
          format strings, providing a mechanism for fields with print
          formats that depart from the standard format
        """

        self.IterationStats = collections.namedtuple('IterationStats', isfld)
        self.isxmap = isxmap
        self.evlmap = evlmap
        self.hdrtxt = hdrtxt
        self.hdrmap = hdrmap

        # Call utility function to construct status display formatting
        self.hdrstr, self.fmtstr, self.nsep = common.solve_status_str(
            hdrtxt, fmtmap=fmtmap, fwdth0=type(self).fwiter,
            fprec=type(self).fpothr)



    def iterstats(self, j, t, isx, evl):
        """Construct IterationStats namedtuple from X step list
        IterationStats namedtuples.

        Parameters
        ----------
        j : int
          Iteration number
        t : float
          Iteration time
        isx : namedtuple
          IterationStats namedtuple from X step object
        evl : dict
          Dict associating result labels with values computed by
          :meth:`DictLearn.evaluate`
        """

        vlst = []
        # Iterate over the fields of the IterationStats namedtuple
        # to be populated with values. If a field name occurs as a
        # key in the isxmap dictionary, use the corresponding key
        # value as a field name in the isx namedtuple for the X
        # step object and append the value of that field as the
        # next value in the IterationStats namedtuple under
        # construction. There are also two reserved field
        # names, 'Iter' and 'Time', referring respectively to the
        # iteration number and run time of the dictionary learning
        # algorithm.
        for fnm in self.IterationStats._fields:
            if fnm in self.isxmap:
                vlst.append(getattr(isx, self.isxmap[fnm]))
            elif fnm in self.evlmap:
                vlst.append(evl[fnm])
            elif fnm == 'Iter':
                vlst.append(j)
            elif fnm == 'Time':
                vlst.append(t)
            else:
                vlst.append(None)

        return self.IterationStats._make(vlst)



    def printheader(self):
        """Print status display header and separator strings."""

        print(self.hdrstr)
        self.printseparator()



    def printseparator(self):
        "Print status display separator string."""

        print("-" * self.nsep)



    def printiterstats(self, itst):
        """Print iteration statistics.

        Parameters
        ----------
        itst : namedtuple
          IterationStats namedtuple as returned by :meth:`iterstats`
        """

        itdsp = tuple([getattr(itst, self.hdrmap[col]) for col in self.hdrtxt])
        print(self.fmtstr % itdsp)



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

        # Check method
        if self.xmethod.lower() != 'convbpdn' and self.xmethod.lower() !='convelasticnet':
            raise ValueError('Parameter xmethod accepted values are: ''ConvBPDN'' '
                                'and ''ConvElasticNet''')

        # Infer outer problem dimensions
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Parse mu
        if 'mu' in kwargs:
            mu = kwargs['mu']
        else:
            mu = [None] * self.cri.dimN

        # Parse lmbda and opt
        if lmbda is None: lmbda =  [None] * self.cri.dimN
        if optx is None: optx =  [None] * self.cri.dimN

        # Parse isc
        if 'isc' in kwargs:
            isc = kwargs['isc']
        else:
            isc = None

        # Decomposed Kruskal Initialization
        self.R = R
        self.Z = list()
        for i,Nvi in enumerate(self.cri.Nv):                    # Ui
            self.Z.append(np.random.randn(Nvi,np.sum(self.R)))

        # Store arguments
        self.S = S
        self.setdict(D0)
        self.lmbda = lmbda
        self.optx = optx
        self.mu = mu

        # Init isc
        if isc is None:

            isc_lst = []                       # itStats from block-solver
            isc_fields = []
            for i = in range(self.cri.dimN):
                str_i = '_{0!s}'.format(i)

                isc_i = IterStatsConfig(
                    isfld=['ObjFun'+str_i, 'PrimalRsdl'+str_i,'DualRsdl'+str_i,
                            'Rho'+str_i],
                    isxmap={'ObjFun'+str_i: 'ObjFun', 'PrimalRsdl'+str_i: 'PrimalRsdl',
                            'DualRsdl'+str_i: 'DualRsdl', 'Rho'+str_i: 'Rho'},
                    evlmap={},
                    hdrtxt=['Fnc'+str_i, 'r'+str_i, 's'+str_i, u('ρ'+str_i)],
                    hdrmap={'Fnc'+str_i: 'ObjFun'+str_i, 'r'+str_i: 'PrimalRsdl'+str_i,
                            's'+str_i: 'DualRsdl'+str_i, u('ρ'+str_i): 'Rho'+str_i}
                )
                isc_fields += isc_i._fields

                isc_lst.append(isc_i)

            # isc_it = IterStatsConfig(       # global itStats  -> No, to be managed in dictlearn
            #     isfld=['Iter','Time'],
            #     isxmap={},
            #     evlmap={},
            #     hdrtxt=['Itn'],
            #     hdrmap={'Itn': 'Iter'}
            # )
            #
            # isc_fields += isc_it._fields

        self.isc_lst = isc_lst
        # self.isc_it = isc_it
        self.isc = collections.namedtuple('IterationStats', isc_fields)

        # Required because dictlrn.DictLearn assumes that all valid
        # xstep objects have an IterationStats attribute
        # self.IterationStats = self.xstep.IterationStats

        self.itstat = []


    def solve(self):
        """Call the solve method of the inner cbpdn object and return the
        result.
        """

        self.xstep = []
        
        itst = []
        for l in range(self.cri.dimN):

            D0c = self.convolvedict(l+1)                # convolvedict
            Sl = self.reshapesignal(l+1)                # reshapesignal

            # Construct 1-dim xstep
            if self.xmethod.lower() == 'convbpdn':      # ConvBPDN
                self.xstep.append(cbpdn.ConvBPDN(D0c, Sl, self.lmbda[l], self.optx[l],
                    dimK=self.cri.dimK, dimN=1))
            else:                                       # ConvElasticNet
                self.xstep.append(cbpdn.ConvElasticNet(D0c, Sl, self.lmbda[l], self.mu[l],
                    self.optx[l], dimK=self.cri.dimK, dimN=1))

            # Solve
            self.xstep[l].solve()

            # Post x-step
            self.Z(l) = self.xstep[l].getcoef()         # Update Kruskal

            # IterationStats
            xitstat = self.xstep.itstat[-1] if self.xstep.itstat else \
                      self.xstep.IterationStats(
                          *([0.0,] * len(self.xstep.IterationStats._fields)))

            itst += self.isc_lst[l].iterstats(0, 0, xitstat, 0) # Accumulate

        self.itstat.append(self.isc(*itst))      # Cast to global itstats and store


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
        """Get iteration stats."""

        return self.itstat
