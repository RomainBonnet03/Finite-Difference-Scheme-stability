# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:19:41 2024

@author: Romain Bonnet-Eymard
"""
# Python libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Personal libraries
from FiniteDifferenceScheme import *
from Boundaries import *


''' Exemple of semi-groupe instabilities for a homogenous Neumann boundary condition of order 2 
at the right boundary '''

''' Spectral properties '''
xi_K    = np.array([0, 0.3*np.pi, 0.8*np.pi])
F_K     = np.array([1,1, 1])                              # All the spatial modes are
                                                          # associated to z = 1
C_K = np.array([1,-1, 1])

scheme  = FiniteDifferenceScheme(xi_K, F_K, C_K)          # Initialize the objet
scheme.schemeGenerator()                                  # Compute the coefficients

''' Draw symbol and check if it is Cauchy-stable '''
scheme.drawSymbol()

scheme.isCauchyStable()

''' Draw the roots of the caracteristic polynomial for z = 1 '''
scheme.drawRoots(1)

''' Compute the global-reflexion matrix and check if it is semi-group stable '''
bound_r  = Neumann(kb=2)

bound_l  = Dirichlet()

scheme.is_SemiGroup_stable(1, bound_r, bound_l, J = 1000)

''' Simulate the scheme over multiple global-reflexions '''
J = 1000
M = scheme.matrixFiniteDifference(J, bound_r, bound_l)  # Compute the finite difference
                                                           # matrix

def u_0(x):                                                # Initial Condition
    return np.exp(-50*(x-1/2)**2)*np.sin(0.8*np.pi*(x-1/2)*J) 

nbr_global_reflexion = 10
N = 2000*nbr_global_reflexion

U        = np.zeros((J+1,N))                                 # Matrix of the solutions at
                                                             # each time step
U_l2     = np.zeros((N,1))                                   # Vector of the l2 norm of the solution
                                                             # at each time step

U[:,0]   = [u_0(x) for x in np.linspace(0,1,J+1)]
U_l2[0,0] = np.sqrt(np.sum(np.abs(U[:,0])**2)/(J+1))

for n in tqdm(range(1,N)):
    U[:, n]    = M.dot(U[:,n-1])
    U_l2[n,0]   = np.sqrt(np.sum(np.abs(U[:,n])**2)/(J+1))


X = np.arange(0,N)
plt.semilogy(X, U_l2, linewidth = 4)
plt.semilogy(X[1500:-1:2000], U_l2[1500:-1:2000], '-r', linewidth = 4)
for j in range(2*nbr_global_reflexion):
      plt.semilogy([500+j*1000, 500+j*1000], [0, np.max(U_l2)], '--k', linewidth = 2)
plt.title('$||U^n||_{l^2}$'+'for Dirichlet/Neumann order {}'.format(2))
plt.xlabel('n')
plt.grid()
plt.ticklabel_format(axis = 'x', style = 'scientific')
plt.show()





