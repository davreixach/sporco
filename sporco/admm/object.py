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
import tensorly as tl
from scipy.linalg import block_diag

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



class GenericConvBPDN(admm.ADMMEqual):
    r"""
    Base class for ADMM algorithm for solving variants of the
    Convolutional BPDN (CBPDN) :cite:`wohlberg-2014-efficient`
    :cite:`wohlberg-2016-efficient` :cite:`wohlberg-2016-convolutional`
    problem.

    |

    .. inheritance-diagram:: GenericConvBPDN
       :parts: 2

    |

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + g( \{ \mathbf{x}_m \} )

    for input image :math:`\mathbf{s}`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps :math:`\mathbf{x}_m`,
    and where :math:`g(\cdot)` is a penalty term or the indicator
    function of a constraint. It is solved via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + g( \{ \mathbf{y}_m \} )
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``Reg`` : Value of regularisation term

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


    class Options(admm.ADMMEqual.Options):
        """ConvBPDN algorithm options

        Options include all of those defined in
        :class:`.admm.ADMMEqual.Options`, together with additional options:

          ``AuxVarObj`` : Flag indicating whether the objective
          function should be evaluated using variable X (``False``) or
          Y (``True``) as its argument. Setting this flag to ``True``
          often gives a better estimate of the objective function, but
          at additional computational cost.

          ``LinSolveCheck`` : Flag indicating whether to compute
          relative residual of X step solver.

          ``HighMemSolve`` : Flag indicating whether to use a slightly
          faster algorithm at the expense of higher memory usage.

          ``NonNegCoef`` : Flag indicating whether to force solution to
          be non-negative.

          ``NoBndryCross`` : Flag indicating whether all solution
          coefficients corresponding to filters crossing the image
          boundary should be forced to zero.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        # Warning: although __setitem__ below takes care of setting
        # 'fEvalX' and 'gEvalY' from the value of 'AuxVarObj', this
        # cannot be relied upon for initialisation since the order of
        # initialisation of the dictionary keys is not deterministic;
        # if 'AuxVarObj' is initialised first, the other two keys are
        # correctly set, but this setting is overwritten when 'fEvalX'
        # and 'gEvalY' are themselves initialised
        defaults.update({'AuxVarObj': False, 'fEvalX': True,
                         'gEvalY': False, 'ReturnX': False,
                         'HighMemSolve': False, 'LinSolveCheck': False,
                         'RelaxParam': 1.8, 'NonNegCoef': False,
                         'NoBndryCross': False})
        defaults['AutoRho'].update({'Enabled': True, 'Period': 1,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              GenericConvBPDN algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)



        def __setitem__(self, key, value):
            """Set options 'fEvalX' and 'gEvalY' appropriately when option
            'AuxVarObj' is set.
            """

            admm.ADMMEqual.Options.__setitem__(self, key, value)

            if key == 'AuxVarObj':
                if value is True:
                    self['fEvalX'] = False
                    self['gEvalY'] = True
                else:
                    self['fEvalX'] = True
                    self['gEvalY'] = False



    itstat_fields_objfn = ('ObjFun', 'DFid', 'Reg')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', 'Reg')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'Reg': 'Reg'}



    def __init__(self, D, S, opt=None, dimK=None, dimN=2):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal),
        `dimN` + 1 dimensional (either multiple channels or multiple
        signals), or `dimN` + 2 dimensional (multiple channels and
        multiple signals). Determination of problem dimensions is
        handled by :class:`.cnvrep.CSC_ConvRepIndexing`.


        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        opt : :class:`GenericConvBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = GenericConvBPDN.Options()

        # Infer problem dimensions and set relevant attributes of self
        if not hasattr(self, 'cri'):
            self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        super(GenericConvBPDN, self).__init__(self.cri.shpX, S.dtype, opt)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.cri.axisN)

        # Initialise byte-aligned arrays for pyfftw
        self.YU = sl.pyfftw_empty_aligned(self.Y.shape, dtype=self.dtype)
        self.Xf = sl.pyfftw_rfftn_empty_aligned(self.Y.shape, self.cri.axisN,
                                                self.dtype)

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), self.rho,
                                      self.cri.axisM)
        else:
            self.c = None



    def getcoef(self):
        """Get final coefficient array."""

        return self.getmin()



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`."""

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho * sl.rfftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.rho, b, self.c,
                                        self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.rho, b, self.cri.axisM,
                                          self.cri.axisC)

        self.X = sl.irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + self.rho * self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.
        If this method is not overridden, the problem is solved without
        any regularisation other than the option enforcement of
        non-negativity of the solution and filter boundary crossing
        supression. When it is overridden, it should be explicitly
        called at the end of the overriding method.
        """

        if self.opt['NonNegCoef']:
            self.Y[self.Y < 0.0] = 0.0
        if self.opt['NoBndryCross']:
            for n in range(0, self.cri.dimN):
                self.Y[(slice(None),) * n +
                       (slice(1 - self.D.shape[n], None),)] = 0.0



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value.
        """

        return self.Xf if self.opt['fEvalX'] else \
            sl.rfftn(self.Y, None, self.cri.axisN)



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m \mathbf{d}_m *
        \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = sl.inner(self.Df, self.obfn_fvarf(), axis=self.cri.axisM) - \
            self.Sf
        return sl.rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def obfn_reg(self):
        """Compute regularisation term(s) and contribution to objective
        function.
        """

        raise NotImplementedError()



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), self.rho,
                                      self.cri.axisM)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.Y
        Xf = sl.rfftn(X, None, self.cri.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)



class ConvBPDN(GenericConvBPDN):
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


    class Options(GenericConvBPDN.Options):
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



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
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
            opt = ConvBPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
            Df = sl.rfftn(D.reshape(cri.shpD), cri.Nv, axes=cri.axisN)
            Sf = sl.rfftn(S.reshape(cri.shpS), axes=cri.axisN)
            b = np.conj(Df) * Sf
            lmbda = 0.1 * abs(b).max()

        # Set l1 term scaling
        self.lmbda = self.dtype.type(lmbda)

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

        # Call parent class __init__
        super(ConvBPDN, self).__init__(D, S, opt, dimK, dimN)

        # Set l1 term weight array
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)
        self.wl1 = self.wl1.reshape(cr.l1Wshape(self.wl1, self.cri))



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.lmbda/self.rho)*np.sign(self.Y)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        self.Y = sp.prox_l1(self.AX + self.U,
                            (self.lmbda / self.rho) * self.wl1)
        super(ConvBPDN, self).ystep()



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        return (self.lmbda*rl1, rl1)



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
        if xmethod.lower() != 'convbpdn' and xmethod.lower() !='convelasticnet':
            raise ValueError('Parameter xmethod accepted values are: ''ConvBPDN'' '
                                'and ''ConvElasticNet''')
        else:
            self.xmethod = xmethod.lower()

        # Infer outer problem dimensions
        # self.cri = cr.CSC_ConvRepIndexing(D0, S, dimK=dimK, dimN=dimN)
        self.cri = cr.CDU_ConvRepIndexing(D0.shape, S, dimK=dimK, dimN=dimN)

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
        self.Z = []
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


    def solve(self):
        """Call the solve method of the inner cbpdn object and return the
        result.
        """

        self.xstep = []

        itst = []
        for l in range(self.cri.dimN):

            print("Kruskal l=%i, preparation: " % int(l+1))

            D0c = self.convolvedict(l+1)                # convolvedict
            Sl = self.reshapesignal(l+1)                # reshapesignal

            # Construct 1-dim xstep
            if self.xmethod == 'convbpdn':              # ConvBPDN
                self.xstep.append(cbpdn.ConvBPDN(D0c, Sl, self.lmbda[l], self.optx[l],
                    dimK=self.cri.dimK, dimN=1))
            else:                                       # ConvElasticNet
                self.xstep.append(cbpdn.ConvElasticNet(D0c, Sl, self.lmbda[l], self.mu[l],
                    self.optx[l], dimK=self.cri.dimK, dimN=1))

            print("Success\nKruskal l=%0i solve time: " % int(l+1))

            # Solve
            self.xstep[l].solve()

            # Post x-step
            self.Z[l] = self.xstep[l].getcoef()         # Update Kruskal

            # IterationStats
            xitstat = self.xstep.itstat[-1] if self.xstep.itstat else \
                      self.xstep.IterationStats(
                          *([0.0,] * len(self.xstep.IterationStats._fields)))

            itst += self.isc_lst[l].iterstats(0, 0, xitstat, 0) # Accumulate

            print("%.2fs" % self.xstep[l].timer.elapsed('solve'), "\n")

        self.itstat.append(self.isc(*itst))      # Cast to global itstats and store


    def setdict(self, D):
        """Set dictionary array."""

        self.D = np.asarray(D)


    def expandZRankDec(self,l=None):
        """Expand Kruskal Zl (Rank Decomposed)."""

        Nv = self.cri.Nv
        R = self.R
        Z = self.Z

        if l is not None:
            Nv = np.delete(Nv,[l-1])            # remove l dimension
            # R = np.delete(R,[l-1])
            Z = np.delete(Z,[l-1])

        Nv = np.append(Nv,np.sum(R))

        Tz = tl.tenalg.khatri_rao(Z)

        return tl.base.vec_to_tensor(Tz,Nv)     # as array Tz(N0,N1,...,K*R)


    def expandZ(self):
        """Expand Kruskal Z."""

        Nv = self.cri.Nv
        R = self.R

        Nv = np.append(Nv,np.sum(R))

        Tzk = np.dot(tl.tenalg.khatri_rao(self.Z),self.getweights())

        return tl.base.vec_to_tensor(Tzk,Nv)     # as array Tz(N0,N1,...,K)


    def convolvedict(self,l=None):
        """~D: Convolve D w/ Zl."""

        axisN = self.cri.axisN
        N = self.cri.N
        Nv = self.cri.Nv
        M = self.cri.M                              # Number of filters K
        dsz = self.cri.dsz                          # Diccionary Size
        Dd = np.prod(dsz[0:-1])

        Tz = self.expandZRankDec(l)

        # TzFull
        Tzlst = []
        for i in range(dsz[l-1]):
            Tzlst.append(Tz)                        # Z for each j

        TzFull = np.stack(Tzlst,axis=l-1)           # As array Tz(N0,N1,...,Nl,...,K*R)

        # DzFull
        DFull = np.reshape(self.D,[Dd,M],order='F')
        DFull = np.dot(DFull,np.transpose(self.getweights()))
        DFull = np.reshape(DFull,np.append(dsz[0:-1],np.sum(self.R)),order='F') # As array D(N0,N1,...,K*R)

        # Purge l
        if l is not None:
            axisN = np.delete(axisN,[l-1])       # remove l dimension
            Nv = np.delete(Nv,[l-1])

        # Convolve
        Xf = sl.rfftn(TzFull,None,axisN)
        Df = sl.rfftn(DFull,Nv,axisN)
        Dcf = np.multiply(Df,Xf)
        Dc = sl.irfftn(Dcf,Nv,axisN)

        Dc = np.moveaxis(Dc,l-1,0)              # spatial dimension at first place

        return np.reshape(Dc,[dsz[l-1],np.prod(Nv),np.sum(self.R)],order='F') # As array Dc(Dl,N',K*R)


    def getweights(self):
        """Linear map from [NxR*K] to [NxK] array."""

        weightsArray = []
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
