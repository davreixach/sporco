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
import collections

import numpy as np
import scipy
from scipy import linalg
import tensorly as tl

from sporco import cdict
from sporco import util
from sporco import common
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


class KCSC_ConvRepIndexing(object):
    """Array dimensions and indexing for Kruskal CSC problems.

    Manage the inference of problem dimensions and the roles of
    :class:`numpy.ndarray` indices for convolutional representations in
    convolutional sparse coding problems (e.g.
    :class:`.admm.cbpdn.ConvBPDN` and related classes).
    """

    def __init__(self, D, S, dimK=None, dimN=2):
        """Initialise a ConvRepIndexing object.

        Initialise a ConvRepIndexing object representing dimensions
        of S (input signal), D (dictionary), and X (coefficient array)
        in a convolutional representation.  These dimensions are
        inferred from the input `D` and `S` as well as from parameters
        `dimN` and `dimK`.  Management and inferrence of these problem
        dimensions is not entirely straightforward because
        :class:`.admm.cbpdn.ConvBPDN` and related classes make use
        *internally* of S, D, and X arrays with a standard layout
        (described below), but *input* `S` and `D` are allowed to
        deviate from this layout for the convenience of the user.

        The most fundamental parameter is `dimN`, which specifies the
        dimensionality of the spatial/temporal samples being
        represented (e.g. `dimN` = 2 for representations of 2D
        images).  This should be common to *input* S and D, and is
        also common to *internal* S, D, and X.  The remaining
        dimensions of input `S` can correspond to multiple channels
        (e.g. for RGB images) and/or multiple signals (e.g. the array
        contains multiple independent images).  If input `S` contains
        two additional dimensions (in addition to the `dimN` spatial
        dimensions), then those are considered to correspond, in
        order, to channel and signal indices.  If there is only a
        single additional dimension, then determination whether it
        represents a channel or signal index is more complicated.  The
        rule for making this determination is as follows:

        * if `dimK` is set to 0 or 1 instead of the default ``None``,
          then that value is taken as the number of signal indices in
          input `S` and any remaining indices are taken as channel
          indices (i.e. if `dimK` = 0 then dimC = 1 and if `dimK` = 1
          then dimC = 0).
        * if `dimK` is ``None`` then the number of channel dimensions is
          determined from the number of dimensions in the input
          dictionary `D`. Input `D` should have at least `dimN` + 1
          dimensions, with the final dimension indexing dictionary
          filters. If it has exactly `dimN` + 1 dimensions then it is a
          single-channel dictionary, and input `S` is also assumed to be
          single-channel, with the additional index in `S` assigned as a
          signal index (i.e. dimK = 1). Conversely, if input `D` has
          `dimN` + 2 dimensions it is a multi-channel dictionary, and
          the additional index in `S` is assigned as a channel index
          (i.e. dimC = 1).

        Note that it is an error to specify `dimK` = 1 if input `S`
        has `dimN` + 1 dimensions and input `D` has `dimN` + 2
        dimensions since a multi-channel dictionary requires a
        multi-channel signal. (The converse is not true: a
        multi-channel signal can be decomposed using a single-channel
        dictionary.)

        The *internal* data layout for S (signal), D (dictionary), and
        X (coefficient array) is (multi-channel dictionary)
        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  C,   1,   M)
          X(N0,  N1, ...,  1,   K,   M)

        or (single-channel dictionary)

        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  1,   1,   M)
          X(N0,  N1, ...,  C,   K,   M)

        where

        * Nv = [N0, N1, ...] and N = N0 x N1 x ... are the vector of sizes
          of the spatial/temporal indices and the total number of
          spatial/temporal samples respectively
        * C is the number of channels in S
        * K is the number of signals in S
        * M is the number of filters in D

        It should be emphasised that dimC and `dimK` may take on values
        0 or 1, and represent the number of channel and signal
        dimensions respectively *in input S*. In the internal layout
        of S there is always a dimension allocated for channels and
        signals. The number of channel dimensions in input `D` and the
        corresponding size of that index are represented by dimCd
        and Cd respectively.

        Parameters
        ----------
        D : array_like
          Input dictionary
        S : array_like
          Input signal
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions of signal samples
        """

        # Determine whether dictionary is single- or multi-channel
        self.dimCd = D.ndim - (dimN + 2)
        if self.dimCd == 0:
            self.Cd = 1
        else:
            self.Cd = D.shape[-2]

        # Numbers of spatial, channel, and signal dimensions in
        # external S are dimN, dimC, and dimK respectively. These need
        # to be calculated since inputs D and S do not already have
        # the standard data layout above, i.e. singleton dimensions
        # will not be present
        if dimK is None:
            rdim = S.ndim - dimN
            if rdim == 0:
                (dimC, dimK) = (0, 0)
            elif rdim == 1:
                dimC = self.dimCd  # Assume S has same number of channels as D
                dimK = S.ndim - dimN - dimC  # Assign remaining channels to K
            else:
                (dimC, dimK) = (1, 1)
        else:
            dimC = S.ndim - dimN - dimK  # Assign remaining channels to C

        self.dimN = dimN  # Number of spatial dimensions
        self.dimC = dimC  # Number of channel dimensions in S
        self.dimK = dimK  # Number of signal dimensions in S

        # Number of channels in S
        if self.dimC == 1:
            self.C = S.shape[dimN]
        else:
            self.C = 1
        Cx = self.C - self.Cd + 1

        # Ensure that multi-channel dictionaries used with a signal with a
        # matching number of channels
        if self.Cd > 1 and self.C != self.Cd:
            raise ValueError("Multi-channel dictionary with signal with "
                             "mismatched number of channels (Cd=%d, C=%d)" %
                             (self.Cd, self.C))

        # Number of signals in S
        if self.dimK == 1:
            self.K = S.shape[self.dimN + self.dimC]
        else:
            self.K = 1

        # Number of filters
        self.M = D.shape[1]     # KCSC std layout D(n',M,n,C)

        # Shape of spatial indices and number of spatial samples
        self.Nv = S.shape[0:dimN]
        self.N = np.prod(np.array(self.Nv))

        self.N_ = D.shape[0:dimN]

        self.nv = tuple(np.array(self.Nv)/self.N_)

        # Axis indices for each component of X and internal S and D
        self.axisN = tuple(range(0, dimN))
        self.axisC = dimN
        self.axisK = dimN + 1
        # self.axisM = dimN + 2

        # Shapes of internal S, D, and X (TO BE DONE, maybe not needed)
        self.shpD = (self.N_,) + (self.M,) + self.nv + (self.Cd,) + (1,)
        self.shpS = self.Nv  + (1,) + (1,) + (self.C,) + (self.K,) + (1,)
        self.shpX = (self.M,) + self.Nv + (Cx,) + (self.K,)



    def __str__(self):
        """Return string representation of object."""

        return pprint.pformat(vars(self))



class KConvBPDN(cbpdn.GenericConvBPDN):
    r"""
    ADMM algorithm for the Convolutional BPDN (CBPDN)
    :cite:`wohlberg-2014-efficient` :cite:`wohlberg-2016-efficient`
    :cite:`wohlberg-2016-convolutional` problem.

    |

    .. inheritance-diagram:: ConvBPDN
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1

    for input image :math:`\mathbf{s}`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps :math:`\mathbf{x}_m`,
    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    Multi-image and multi-channel problems are also supported. The
    multi-image problem is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1

    with input images :math:`\mathbf{s}_k` and coefficient maps
    :math:`\mathbf{x}_{k,m}`, and the multi-channel problem with input
    image channels :math:`\mathbf{s}_c` is either

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
       \mathbf{s}_c \right\|_2^2 +
       \lambda \sum_c \sum_m \| \mathbf{x}_{c,m} \|_1

    with single-channel dictionary filters :math:`\mathbf{d}_m` and
    multi-channel coefficient maps :math:`\mathbf{x}_{c,m}`, or

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -
       \mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1

    with multi-channel dictionary filters :math:`\mathbf{d}_{c,m}` and
    single-channel coefficient maps :math:`\mathbf{x}_m`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """


    class Options(cbpdn.GenericConvBPDN.Options):
        r"""ConvBPDN algorithm options

        Options include all of those defined in
        :class:`.admm.ADMMEqual.Options`, together with additional options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the `X`/`Y` variables (see
          :func:`.cnvrep.l1Wshape` for more details). If this
          option is defined, the regularization term is :math:`\lambda
          \sum_m \| \mathbf{w}_m \odot \mathbf{x}_m \|_1` where
          :math:`\mathbf{w}_m` denotes slices of the weighting array on
          the filter index axis.
        """

        defaults = copy.deepcopy(GenericConvBPDN.Options.defaults)
        defaults.update({'L1Weight': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDN algorithm options
            """

            if opt is None:
                opt = {}
            GenericConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegL2')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2'): 'RegL2'}


    def __init__(self, Df, Sf, lmbda=None, mu=0.0 opt=None, dimK=None, dimN=1):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal), `dimN`
        + 1 dimensional (either multiple channels or multiple signals),
        or `dimN` + 2 dimensional (multiple channels and multiple
        signals). Determination of problem dimensions is handled by
        :class:`.cnvrep.CSC_ConvRepIndexing`.


        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdn_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdn_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`ConvBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = KConvBPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, Sf.dtype)

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CSC_ConvRepIndexing(Df, Sf, dimK=dimK, dimN=dimN)

        # Reshape D and S to standard layout
        self.Df = np.asarray(Df.reshape(self.cri.shpD), dtype=self.dtype)
        self.Sf = np.asarray(Sf.reshape(self.cri.shpS), dtype=self.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            b = np.conj(Df) * Sf
            lmbda = 0.1 * abs(b).max()

        # Set l1 term scaling
        self.lmbda = self.dtype.type(lmbda)

        # Set l2 term scaling
        self.mu = self.dtype.type(mu)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0 * self.lmbda + 1.0),
                      dtype=self.dtype)

        # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
        if self.lmbda != 0.0:
            rho_xi = float((1.0 + (18.3)**(np.log10(self.lmbda) + 1.0)))
        else:
            rho_xi = 1.0
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=rho_xi,
                      dtype=self.dtype)

        # Call parent class __init__ (not ConvBPDN bc FFT domain data)
        super(cbpdn.GenericConvBPDN, self).__init__(self.cri.shpX, Sf.dtype, opt)

        # Initialise byte-aligned arrays for pyfftw
        self.YU = sl.pyfftw_empty_aligned(self.Y.shape, dtype=self.dtype)
        self.Xf = sl.pyfftw_rfftn_empty_aligned(self.Y.shape, self.cri.axisN,
                                                self.dtype)

        self.setdict()

        # Not 'HighMemSolve' for this class
        self.c = None

        # Set l1 term weight array
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)
        self.wl1 = self.wl1.reshape(cr.l1Wshape(self.wl1, self.cri))


    def setdictf(self, Df=None):
        """Set dictionary array."""

        if Df is not None:
            self.Df = Df;
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)

        for n in range(self.cri.N)
        self.c = linalg.cho_factor(np.dot(self.DSf,self.Df),
            (self.mu + self.rho)* np.identity(self.cri.M,dtype=self.dtype),lower=False,check_finite=True)


    def getcoeff(self):
        """Get final coefficient array (FFT domain)."""

        return self.getmin()


    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.lmbda/self.rho)*np.sign(self.Y)


    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho*sl.rfftn(self.YU, None, self.cri.axisN)
        # if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.mu + self.rho,
                                        b, self.c, self.cri.axisM)
            self.Xf[:] = sl.cho_solve_ATAI(self.Df,self.mu + self.rho,
                                        b, self.c,False)
        # else:
        #     self.Xf[:] = sl.solvemdbi_ism(self.Df, self.mu + self.rho, b,
        #                                   self.cri.axisM, self.cri.axisC)

        # self.X = sl.irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + (self.mu + self.rho)*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None


    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        self.Y = sp.prox_l1(self.AX + self.U,
                            (self.lmbda / self.rho) * self.wl1)
        super(cbpdn.ConvBPDN, self).ystep()


    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function. (ConvElasticNet)
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl2 = 0.5*np.linalg.norm(self.obfn_gvar())**2
        return (self.lmbda*rl1 + self.mu*rl2, rl1, rl2)


    def setdict(self):
        """Set dictionary array.

        Overriding this method is required.
        """

        raise NotImplementedError()


    def getcoef(self):
        """Get final coefficient array.

        Overriding this method is required.
        """

        raise NotImplementedError()


    def reconstruct(self):
        """Reconstruct representation.

        Overriding this method is required.
        """

        raise NotImplementedError()


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

    class Options(cdict.ConstrainedDict):
        """AKConvBPDN options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``StatusHeader`` : Flag determining whether status header and
          separator are displayed.

          ``IterTimer`` : Label of the timer to use for iteration times.

          ``MaxMainIter`` : Maximum main iterations.

          ``Callback`` : Callback function to be called at the end of
          every iteration.
        """

        defaults = {'Verbose': False, 'StatusHeader': True,
                    'IterTimer': 'solve', 'MaxMainIter': 50,
                    'Callback': None}


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              DictLearn algorithm options
            """

            if opt is None:
                opt = {}
            cdict.ConstrainedDict.__init__(self, opt)


    def __new__(cls, *args, **kwargs):
        """Create an AKConvBPDN object and start its
        initialisation timer."""

        instance = super(AKConvBPDN, cls).__new__(cls)
        instance.timer = util.Timer(['init', 'solve', 'solve_wo_eval'])
        instance.timer.start('init')
        return instance


    def __init__(self, D0, S, R, opt=None, lmbda=None, optx=None,
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

        if opt is None:
            opt = AKConvBPDN.Options()
        self.opt = opt

        # Infer outer problem dimensions
        self.cri = cr.CSC_ConvRepIndexing(D0.shape, S, dimK=dimK, dimN=dimN)

            # Parse mu
            if 'mu' in kwargs:
                mu = kwargs['mu']
            else:
                mu = [0] * self.cri.dimN

            # Parse lmbda and optx
            if lmbda is None: lmbda =  [None] * self.cri.dimN
            if optx is None: optx =  [None] * self.cri.dimN

            # Parse isc
            if 'isc' in kwargs:
                isc = kwargs['isc']
            else:
                isc = None

        # Decomposed Kruskal Initialization
        self.R = R
        self.Kf = []
        for i,Nvi in enumerate(self.cri.Nv):                    # Ui
            self.Kf.append(np.random.randn(Nvi,np.sum(self.R)))

        self.K = [None]*self.cri.dimN

        # Store parameters
        self.lmbda = lmbda
        self.optx = optx
        self.mu = mu

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.cri.axisN)

        # Signal uni-dim (kruskal)
        self.Skf = np.reshape(self.Sf,[S.size,1],order='F')

        self.setdict()

        # Init KCSC solver (Needs to be initiated inside AKConvBPDN because requires convolvedict() and reshapesignal())
        self.xstep = []
        for l in range(self.cri.dimN):

            Wl = self.convolvedict(l)                # convolvedict

            self.xstep.append(KConvBPDN(Wl, self.Skf, self.lmbda[l], self.optx[l],
                    dimK=self.cri.dimK, dimN=1))

        # Init isc
        if isc is None:

            isc_lst = []                       # itStats from block-solver
            isc_fields = []
            for i in range(self.cri.dimN):
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
                isc_fields += isc_i.IterationStats._fields

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
        self.j = 0


    def solve(self):
        """Call the solve method of the inner KConvBPDN object and return the
        result.
        """

        itst = []

        # Main optimisation iterations
        for self.j in range(self.j, self.j + self.opt['MaxMainIter']):

            for l in range(self.cri.dimN):

                # Pre x-step
                Wl = self.convolvedict(l)                # convolvedict
                self.xstep[l].setdictf(Wl)               # setdictf

                # Solve KCSC
                self.xstep[l].solve()

                # Post x-step
                self.Kf[l] = self.xstep[l].getcoeff()         # Update Kruskal

                # IterationStats
                xitstat = self.xstep.itstat[-1] if self.xstep.itstat else \
                          self.xstep.IterationStats(
                              *([0.0,] * len(self.xstep.IterationStats._fields)))

                itst += self.isc_lst[l].iterstats(self.j, 0, xitstat, 0) # Accumulate

            self.itstat.append(self.isc(*itst))      # Cast to global itstats and store

        # Decomposed ifftn
        for l in range(self.cri.dimN):
            self.K[l] = sl.irfftn(self.Kf[l], self.cri.Nv[l], 0) # ifft transform

        self.j += 1


    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)

    def convolvedict(self,l=None):
        """W: Convolve D w/         # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.cri.axisN)

        # Signal uni-dim (kruskal)
        self.Skf = np.reshape(self.Sf,[S.size,1],order='F')

        self.setdict()Zl."""

        Df = self.Df
        Kf = self.Kf
        N = self.cri.N

        Nl = Nv[l] if l is not None else 1

        Df_ = np.moveaxis(np.reshape(Df,[Nl,N/Nl],order='F'),[0,1,2],[0,2,1]).squeeze()
        Q = tl.tenalg.khatri_rao(Kf,skip_matrix=l,reverse=True)
        return Df_*Q


    def getweights(self):
        """Linear map from [NxR*K] to [NxK] array."""

        weightsArray = []
        for k,Rk in enumerate(self.R):
            weightsArray.append(np.ones([Rk,1]))

        return sl.block_diag(*weightsArray)                 # map from R*M to M


    def getcoef(self):
        """Get result of inner xstep object and expand Kruskal."""

        Nz = self.cri.Nv
        Nz.append(self.cri.M)

        Z = np.dot(tl.tenalg.khatri_rao(self.K,reverse=True),self.getweights())

        return tl.base.vec_to_tensor(Z,Nz)     # as array Z(N0,N1,...,M)


    def getKruskal(self):
        """Get decomposed Krukal Z."""

        return self.K()


    def reconstruct(self, X=None):
        """Reconstruct representation."""

        Df = self.Df
        Nz = self.cri.Nv
        Nz.append(self.cri.M)

        # # Stupid Option
        # Tz = self.getcoef()
        # Xf = sl.rfftn(Tz,None,self.cri.axisN)

        #Smart Option
        Zf = np.dot(tl.tenalg.khatri_rao(self.Kf,reverse=True),self.getweights())
        Xf = tl.base.vec_to_tensor(Z,Nz)

        Sf = np.sum(Df*Xf, axis=self.cri.axisM)

        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)


    def getitstat(self):b
        """Get iteration stats."""

        return self.itstat
