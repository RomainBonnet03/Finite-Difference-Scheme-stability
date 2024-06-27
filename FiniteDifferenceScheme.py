#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:05:44 2024

@author: romainbonnet-eymard
"""

# Python Libraries #
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as nppol

# Personal Libraries #
from Lebarbenchon_codes.utils import sort, epsilon, eta_func
from Boundaries import *

class FiniteDifferenceScheme:
    
    def __init__(self, xi_liste, F_liste, C_liste):
        # The lists must be of type (N,0)-ndarray
        self.xi      = xi_liste
        self.F       = F_liste
        self.C       = C_liste
        self.stencil = 0
        self.center  = 0
        self.r       = 0
        self.p       = 0
        self.exists  = False                   # True if there exists a scheme satisfying the spectral 
                                               # properties
                                               
        self.is_Cauchy_stable = False          # True if the scheme (if it exists) satisfies the
                                               # Von-Neumann 
        
    def schemeGenerator(self):
        ''' Determine the stencil and the coefficient of the scheme which satisfies the
         imposed spectral properties. Carefull, the system might be singular! '''
        
        K     = np.size(self.xi)               # Number of imposed spatial frequencies
        kappa = np.exp(1j*self.xi)             # Spatial mod associated to the spatial frequencies xi
        
        ''' Count the number of non real kappa '''
        treshold = 10**(-15)
        compte   = 0
        
        for k in range(K):
            if np.abs(np.imag(kappa[k]))>treshold:
                compte += 1
            
        ''' Assembly of the system '''
        stencil  = 3*(K+compte)                 # Each complex kappa multiplies by 2 the number 
                                                # of constraints
        
        A        = np.zeros((stencil,stencil))  # Matrix encoding the system
        b        = np.zeros((stencil,1))        # Left-hand side
        
        ind_cour = 0                             
        
        for k in range(K):
            
            # Series expansion of the symbol F, near xi_K, at order 0 
            ordre_0    = [kappa[k]**(l-stencil//2) for l in range(stencil)]                   
            # Series expansion of the symbol F, near xi_K, at order 1
            ordre_1    = [(l-stencil//2)*kappa[k]**(l-stencil//2) for l in range(stencil)]    
            # Series expansion of the symbol F, near xi_K, at order 2
            ordre_2    = [(l-stencil//2)**2*kappa[k]**(l-stencil//2) for l in range(stencil)] 
            
            if np.abs(np.imag(kappa[k]))<treshold:
                A[ind_cour,:]    = np.real(ordre_0)                   
                A[ind_cour+1,:]  = np.real(ordre_1)
                A[ind_cour+2,:]  = np.real(ordre_2)
                
                b[ind_cour, 0]   = np.real(self.F[k])
                b[ind_cour+1, 0] = np.real(-self.F[k]*self.C[k])
                b[ind_cour+2, 0] = np.real(self.F[k]*(self.C[k]**2+1))
                
                ind_cour += 3
        
            else:
                A[ind_cour,:]    = np.real(ordre_0)
                A[ind_cour+1,:]  = np.imag(ordre_0)
                A[ind_cour+2,:]  = np.real(ordre_1)
                A[ind_cour+3,:]  = np.imag(ordre_1)
                A[ind_cour+4,:]  = np.real(ordre_2)
                A[ind_cour+5,:]  = np.imag(ordre_2)
                
                b[ind_cour, 0]   = np.real(self.F[k])
                b[ind_cour+1, 0] = np.imag(self.F[k])
                b[ind_cour+2, 0] = np.real(-self.F[k]*self.C[k])
                b[ind_cour+3, 0] = np.imag(-self.F[k]*self.C[k])
                b[ind_cour+4, 0] = np.real(self.F[k]*(self.C[k]**2+1))
                b[ind_cour+5, 0] = np.imag(self.F[k]*(self.C[k]**2+1))
                
                ind_cour += 6
                
        if np.linalg.matrix_rank(A)==stencil:    # Verify the system is not singular
            self.exists  = True
            
            coeff        = np.linalg.solve(A,b)
            while np.abs(coeff[0,0])<10**(-12):  # Pop the first and last coefficients in case they are
                                                 # equal to 0
                coeff = coeff[1:,:]
            while np.abs(coeff[-1,0])<10**(-12):
                coeff = coeff[:-1,:]
            
            self.scheme  = coeff
            self.stencil = np.size(coeff)
            self.center  = self.stencil//2
            self.r       = self.center
            self.p       = self.stencil - self.center - 1

        
    def symbol(self, theta):
        ''' Compute the symbol of the scheme, at the frequency xi, namely the mod exp(1j*xi) '''

        assert(self.exists)
        result = 0
        for l in range(self.stencil):
            j = l-self.stencil//2
            result += self.scheme[l,0]*np.exp(1j*j*theta)
        return result
    
    def isCauchyStable(self, treshold = 10**(-10)):
        assert(self.exists)
        theta_liste = np.linspace(0,2*np.pi,10**2)
        F = self.symbol(theta_liste)
        self.NeumannStable = np.sum(np.abs(F)-1>treshold) == 0

    def drawSymbol(self):
        ''' Plot the symbol of the scheme on the unit circle '''

        assert(self.exists)
        Theta           = np.linspace(-np.pi, np.pi, 10**3)
        spectre      = self.symbol(Theta)
        cercle_unite = np.exp(1j*Theta)

        plt.plot(np.real(spectre), np.imag(spectre))
        plt.plot(np.real(cercle_unite), np.imag(cercle_unite), '--r')
        plt.plot([-1,1],[0,0],'k')
        plt.plot([0,0], [-1,1],'k')
        plt.axis([-2,2,-2,2])
        plt.show()
        
    def _pol(self, z):
        ''' Credit: Pierre Le Barbenchon '''
        ''' Compute the coefficient of the caracteristic polynomial associated to the scheme,
         at the temporal mod z '''
        
        assert(self.exists)
        r = self.center
        monome = nppol.Polynomial([0, 1])
        P = nppol.Polynomial(self.scheme[:,0]) - z * monome**r
        return P
    
    def _roots(self, z):
        ''' Credit: Pierre Le Barbenchon '''
        ''' Compute the roots of the caracteristic polynomial associated to the scheme,
         at the temporal mod z '''
        
        assert(self.exists)
        Racinestotales = self._pol(z).roots()
        return sort(Racinestotales)

    def _count_root(self, eta, eps, z0, kappa):
        ''' Credit: Pierre Le Barbenchon '''
        ''' Compute the number of forward mods and the number of backward mods '''
        
        assert(self.exists)
        z = z0 + eta * z0 / (2 * abs(z0))
        NewRoots  = self._roots(z)
        selection = list(filter(lambda k: abs(k - kappa) < eps, NewRoots))
        n_forward  = len(list(filter(lambda k: abs(k) < 1, selection)))
        n_backward = len(list(filter(lambda k: abs(k) > 1, selection)))
        return (n_forward, n_backward)
    
    def _Kappa(self, z0):
        """
        Credit: Pierre Le Barbenchon
        sort the roots according to either they are forward or backward
        """
        assert(self.exists)
        delta = 10 ** (-10)
        Racinestotales = self._roots(z0)
        eps = epsilon(Racinestotales)
        RootsFromInside  = []
        RootsFromOutside = []
        stock = []
        for x in Racinestotales:
            if abs(x) < 1 - delta:
                RootsFromInside.append(x)
            elif abs(x) < 1 + delta and x not in stock:
                stock.append(x)
                eta = eta_func(eps, x, 1000, self._pol(z0), self.center)
                [n_inside, n_outside] = self._count_root(eta, eps, z0, x)
                for i in range(n_inside):
                    RootsFromInside.append(x)
                for i in range(n_outside):
                    RootsFromOutside.append(x)
            else:
                RootsFromOutside.append(x)
        
        assert len(RootsFromInside) == self.center
        
        return (RootsFromInside, RootsFromOutside)

    def drawRoots(self,z0):
        ''' Scatter the roots of the caracteristic polynomial associated to the scheme, at
        the temporal mod z'''
        assert(self.exists)
        [RootsFromInside, RootsFromOutside] = self._Kappa(z0)


        theta = np.linspace(0,2*np.pi,10**3)
        plt.plot(np.cos(theta), np.sin(theta), 'k')
        plt.scatter(np.real(RootsFromInside), np.imag(RootsFromInside), 30, 'b', label='forward')
        plt.scatter(np.real(RootsFromOutside), np.imag(RootsFromOutside), 30, 'r', label='backward')
        plt.legend()
        plt.title('Racine de P(X) pour z0={}'.format(z0))
        plt.show()
     
    
    def _modalBase(self, z0, bound_r, bound_l):
        assert(self.exists and self.isCauchyStable)
        [RootsFromInside, RootsFromOutside] = self._Kappa(z0)
        
        K = np.zeros((self.p+self.r+bound_r.m+bound_l.m, self.r+self.p), dtype='cfloat')
        
        for k in range(self.p+bound_r.m):
                j = 0
                while j<self.p:
                    mult = 0
                    while j+mult < self.p and RootsFromOutside[j] == RootsFromOutside[j+mult]:
                        K[k,j+mult] = (self.p-k)**mult*RootsFromOutside[j+mult]**(self.p-k)
                        mult+=1
                    j = j+mult
                    
        
                j = 0
                while j<self.r:
                    mult = 0
                    while j+mult < self.r and RootsFromInside[j] == RootsFromInside[j+mult]:
                        K[k,self.p+j+mult] = (self.p-k)**mult*RootsFromInside[j+mult]**(self.p-k)
                        mult+=1
                    j = j+mult
                    
        for k in range(self.r+bound_l.m):   
                kt = k+self.p+bound_r.m
                j = 0
                while j<self.p:
                    mult = 0
                    while j+mult < self.p and RootsFromOutside[j] == RootsFromOutside[j+mult]:
                        K[kt,j+mult] = (-self.r+k)**mult*RootsFromOutside[j+mult]**(-self.r+k)
                        mult+=1
                    j = j+mult
                
                j = 0
                while j<self.r:
                    mult = 0
                    while j+mult < self.r and RootsFromInside[j] == RootsFromInside[j+mult]:
                        K[kt,self.p+j+mult] = (-self.r+k)**mult*RootsFromInside[j+mult]**(-self.r+k)
                        mult+=1
                    j = j+mult
        return K
    
    def matrixReflexion(self, z0, bound_r, bound_l, J):
        ''' Compute the reflexion Matrix associated to the scheme + boundary_g + boundary_d 
        for the temporal mod z0 '''

        assert(self.exists and self.isCauchyStable)
        [RootsFromInside, RootsFromOutside] = self._Kappa(z0)
        
        K = self._modalBase(z0, bound_r, bound_l)
 
        ''' Right Boundary '''
        bound_r(self.p)    # Assemble the B matrix
        B_r = bound_r.B
        
        K_right_d = K[0:self.p+bound_r.m, self.p:]
        K_left_d  = K[0:self.p+bound_r.m, 0:self.p]
        
        Matrix_r = np.linalg.inv(B_r.dot(K_left_d)).dot(-B_r.dot(K_right_d))
                            
        ''' Left Boundary '''
        bound_l(self.r)    # Assemble the B matrix
        B_l = bound_l.B
        K_right_g = K[self.p+bound_r.m:, self.p:]
        K_left_g  = K[self.p+bound_r.m:, 0:self.p]
        
        Matrix_l = np.linalg.inv(B_l.dot(K_right_g)).dot(-B_l.dot(K_left_g))
    
        ''' Compute the D_p^(-J) and D_r^J matrices '''
        Dp = np.zeros((self.p,self.p), dtype = 'cfloat')
        for j in range(self.p):
            Dp[j,j] = RootsFromOutside[j]**(-(J))
        
        Dr = np.zeros((self.r,self.r), dtype = 'cfloat')
        for j in range(self.r):
            Dr[j,j] = RootsFromInside[j]**(J) 
    
        return Matrix_l.dot(Dp.dot(Matrix_r.dot(Dr)))
        
    def is_SemiGroup_stable(self, z0, bound_r, bound_l, J):
        R = self.matrixReflexion(z0, bound_r, bound_l, J)
        rho = np.max(np.abs(np.linalg.eigvals(R)))
        
        if abs(rho)>1:
            print('The scheme IS NOT semi-group stable // rho = ', rho)
        else:
            print('The scheme IS semi-group stable')
        
    def matrixFiniteDifference(self,J, bound_r, bound_l):
        bound_r(self.p)    # Assemble the B matrix
        bound_l(self.r)    # Assemble the B matrix
        
        B_r = bound_r.B    
        B_l = bound_l.B    


        M = np.zeros((J+1,J+1))
        for j in range(0,J+1):
            for l in range(self.stencil):
                lt = l-self.stencil//2
        
                if not(j+lt<0 or j+lt>=J+1):  # Correspond to the interior
                    M[j,j+lt] += self.scheme[l,0]
        
                elif j+lt>=J+1:               # Correspond to the right boundary condition
                    c = j+lt-J
                    for k in range(bound_r.m):
                        M[j,J-k]  += -B_r[self.p-c,self.p+k]*self.scheme[l,0]
                
                else:                         # Correspond to the left boundary condition
                    c = j+lt
                    for k in range(bound_l.m):
                        M[j,k]  += -B_l[self.r+c-1,self.r+k]*self.scheme[l,0]

        return M
        