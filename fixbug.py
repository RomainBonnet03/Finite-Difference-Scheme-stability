# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:22:37 2024

@author: dutym
"""
from FiniteDifferenceScheme import *
from Boundaries import *

xi_K    = np.array([0, 0.3*np.pi, 0.8*np.pi])
F_K     = np.array([1,1, 1])
alpha_K = np.array([1,-1, 1])

Schema  = FiniteDifferenceScheme(xi_K, F_K, alpha_K)
Schema.schemeGenerator()

bound_r = Neumann()
bound_r(Schema.p, 2)

bound_l = Dirichlet()
bound_l(Schema.r)

Schema.isCauchyStable()
print(np.max(np.abs(np.linalg.eigvals(Schema.matrixReflexion(1, bound_r, bound_l, 10**3)))))