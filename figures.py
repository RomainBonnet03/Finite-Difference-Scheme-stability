# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:30:25 2024

@author: Romain Bonnet-Eymard

This script enables to draw the figures of the manuscript.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from FiniteDifferenceScheme import *
import Boundaries as Bd

from Lebarbenchon_codes.complex_winding_number import *
import Lebarbenchon_codes.schemes as sch
import Lebarbenchon_codes.boundaries as bd

########################################
### Figure 1.3 : Lax-Wendroff symbol ###
########################################

def symbol_LW(lambd, theta):
    a_1 = (lambd+lambd**2)/2
    a0  = (1-lambd**2)
    a1  = (-lambd+lambd**2)/2
    
    return a_1*np.exp(-1j*theta) + a0 + a1*np.exp(1j*theta)

Theta = np.linspace(0,2*np.pi,10**3)

plt.rcParams.update({'font.size': 30})

plt.plot(np.real(symbol_LW(0.7, Theta)), np.imag(symbol_LW(0.7, Theta)), label='λ=0.7', linewidth=5)
plt.plot(np.real(symbol_LW(0.9, Theta)), np.imag(symbol_LW(0.9, Theta)), label='λ=0.9', linewidth=5)
plt.plot(np.real(symbol_LW(1, Theta)),np.imag(symbol_LW(1, Theta)), label='λ=1', linewidth=5)
plt.plot(np.real(symbol_LW(1.1, Theta)), np.imag(symbol_LW(1.1, Theta)), label='λ=1.1', linewidth=5)
plt.plot(np.cos(Theta), np.sin(Theta), '--k', linewidth=5)
plt.plot([-1.5, 1.5], [0,0], 'k', linewidth=5)
plt.plot([0,0], [-1.5, 1.5], 'k', linewidth=5)
plt.xlim([-1.7,1.7])
plt.ylim([-1.7,1.7])
plt.legend()
plt.axis('equal')
plt.show()


### Figure 1.5: Kreiss-Lopatinskii determinant ### 
lambd_list = [0.3, 0.5, 0.7]

# Dirichlet : left boundary #
for lambd in lambd_list:
    schema   = sch.LaxWendroff(lamb = lambd, boundary = bd.Dirichlet())

    n_param = 10**3
    parametrization_bool = True

    fDKL = schema.DKL()

    param, curve = schema.detKL(n_param, fDKL, parametrization_bool)

    plt.plot(np.real(curve), np.imag(curve), linewidth=3, label='$\Delta(\mathbb{S})$')
    
    
plt.plot([np.min(np.real(curve)), np.max(np.real(curve))], [0,0], 'k')
plt.plot([0,0], [np.min(np.imag(curve)), np.max(np.imag(curve))], 'k')
plt.legend()
plt.ylabel('Im')
plt.xlabel('Re')
plt.title('Dirichlet à gauche')
plt.show()

# Neumann : left boundary #
for lambd in lambd_list:
    schema   = sch.LaxWendroff(lamb = lambd, boundary = bd.Neumann(1))

    n_param = 10**3
    parametrization_bool = True

    fDKL = schema.DKL()

    param, curve = schema.detKL(n_param, fDKL, parametrization_bool)

    plt.plot(np.real(curve), np.imag(curve), linewidth=3, label='$\Delta(\mathbb{S})$')
    
    
plt.plot([np.min(np.real(curve)), np.max(np.real(curve))], [0,0], 'k')
plt.plot([0,0], [np.min(np.imag(curve)), np.max(np.imag(curve))], 'k')
plt.legend()
plt.ylabel('Im')
plt.xlabel('Re')
plt.title('Dirichlet à gauche')
plt.show()

# Neumann : right boundary #
for lambd in lambd_list:
    schema   = sch.LaxWendroffInversed(lamb = lambd, boundary = bd.Neumann(1))

    n_param = 10**3
    parametrization_bool = True

    fDKL = schema.DKL()

    param, curve = schema.detKL(n_param, fDKL, parametrization_bool)

    plt.plot(np.real(curve), np.imag(curve), linewidth=3, label='$\Delta(\mathbb{S})$')
    
    
plt.plot([np.min(np.real(curve)), np.max(np.real(curve))], [0,0], 'k')
plt.plot([0,0], [np.min(np.imag(curve)), np.max(np.imag(curve))], 'k')
plt.legend()
plt.ylabel('Im')
plt.xlabel('Re')
plt.title('Dirichlet à gauche')
plt.show()

### Figure 1.7: GKS instabilities ### 
T              = 1
lambd          = 0.5
dx             = 1/(100)
dt             = lambd*dx
J              = int(1/dx+1)
N              = int(T/dt+1)

D_LW    = np.zeros((J,J))
D_LW[0,0], D_LW[0,1] = (lambd**2+lambd)/2 + (1-lambd**2), (lambd**2-lambd)/2
D_LW[J-1,J-2], D_LW[J-1,J-1] = (lambd**2+lambd)/2, (1-lambd**2) + (lambd**2-lambd)/2
for j in range(1, J-1):
    D_LW[j,j-1], D_LW[j,j], D_LW[j,j+1] = (lambd**2+lambd)/2, (1-lambd**2), (lambd**2-lambd)/2

U    = np.zeros((J,N))

U[:15,0] = [(-1)**(j) for j in range(15)] # initial condition

for n in range(1,N):
    U[:,n] = D_LW.dot(U[:,n-1])

plt.subplot(3,1,1)
plt.plot(U[:,0], label='t=0')
plt.grid()
plt.legend()
plt.ylabel('$U^n_j$')

plt.subplot(3,1,2)
plt.plot(U[:,50], label='t=0.5')
plt.grid()
plt.legend()
plt.ylabel('$U^n_j$')

plt.subplot(3,1,3)
plt.plot(U[:,99], label='t=1')
plt.grid()
plt.legend()
plt.ylabel('$U^n_j$')
plt.xlabel('j')

plt.show()
    
##############################################
### Example 1.39: Figure 1.9 / 1.10 / 1.11 ### 
##############################################

''' Imposed spectral properties '''
xi_K    = np.array([0, np.pi])
F_K     = np.array([1,1])
C_K = np.array([1,-1])

Schema  = FiniteDifferenceScheme(xi_K, F_K, C_K)
Schema.schemeGenerator()
Schema.isCauchyStable()

# Draw Roots - Figure 1.9 #
z0 = 1 # 1.1
[RootsFromInside, RootsFromOutside] = Schema._Kappa(z0)   # Normally, the Kappa method is private but we make an exeption here (as Python 
                                                          # allows it) to have a cleaner plot of the roots

plt.rcParams.update({'font.size': 30})
theta = np.linspace(0,2*np.pi,10**3)

plt.plot(np.cos(theta), np.sin(theta), 'k', linewidth = 5)

plt.scatter(np.real(RootsFromInside), np.imag(RootsFromInside), s=4**5, c='b', label='rightgoing')
plt.scatter(np.real(RootsFromOutside), np.imag(RootsFromOutside), s=4**5, c='r', label='leftgoing')
plt.legend()
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Racine de P(X) pour z0={}'.format(z0))
plt.show()

# Simulation - Figure 1.10 #

mu_b = 10 # 1.3
mu_a = 10**(-1)

J  = 10**2+1
N  = 100
dx = 1/(J-1)

def u_0(x):
    return np.exp(-50*(x-1/2)**2)*np.cos((x-1/2)*np.pi*J)

M  = np.zeros((J,J))

for j in range(0,J):
   for l in range(Schema.stencil):
       lt = l-Schema.stencil//2

       if not(j+lt<0 or j+lt>=J):
           M[j,j+lt] += Schema.scheme[l,0]

       elif j+lt<0:
           M[j,0]   += (mu_b+mu_a*dx)**(j+lt)*Schema.scheme[l,0]

X = np.linspace(0,1,J)
           
U_RD = np.zeros((J,N))
U_RD[:,0] = u_0(X)

for n in tqdm(range(1,N)):
    U_RD[:, n] = M.dot(U_RD[:,n-1])

plt.rcParams.update({'font.size': 30})
plt.rcParams["font.weight"] = "bold"

plt.subplot(3,1,1)
plt.plot(X,U_RD[:,0],linewidth = 4, label='t=0')
plt.ylabel('$U^n_j$')
plt.legend()
plt.grid()
plt.title(u"$\u03bc_a$=0.1 // $\u03bc_b$=10")

plt.subplot(3,1,2)
plt.plot(X,U_RD[:,40],linewidth = 4,label='t=0.5')
plt.ylabel('$U^n_j$')
plt.legend()
plt.grid()

plt.subplot(3,1,3)
plt.plot(X,U_RD[:,80],linewidth = 4,label='t=0.8')
plt.ylabel('$U^n_j$')
plt.legend()
plt.grid()
plt.xlabel('j')

plt.show()


# Kreiss-Lopatinskii determinant - Figure 1.11 #
''' we have to use Lebarbenchon's code for this part and convert our 'FiniteDifferentScheme' object
to 'scheme' object'''
coeff = []
for j in range(0,Schema.stencil):
    coeff.append(Schema.scheme[j,0])

centre     = Schema.center
schema_1   = sch.Scheme(np.array(coeff), centre, bd.Robin_ordre1(0.1, 1.5, dx))
schema_2   = sch.Scheme(np.array(coeff), centre, bd.Robin_ordre1(0.1, 10, dx))

n_param = 10**3
parametrization_bool = True

fDKL_1 = schema_1.DKL()
fDKL_2 = schema_2.DKL()

param, curve_1 = schema_1.detKL(n_param, fDKL_1, parametrization_bool)
param, curve_2 = schema_2.detKL(n_param, fDKL_2, parametrization_bool)

plt.subplot(1,2,1)
plt.plot(np.real(curve_1), np.imag(curve_1), linewidth=3, label='$\Delta(\mathbb{S})$')
plt.plot([np.min(np.real(curve_1)), np.max(np.real(curve_1))], [0,0], 'k')
plt.plot([0,0], [np.min(np.imag(curve_1)), np.max(np.imag(curve_1))], 'k')
plt.legend()
plt.ylabel('Im')
plt.xlabel('Re')
plt.title(u"$\u03bc_b$=1.5 // r-$Ind_{\Delta \mathbb{S}}=$"+str(schema_1.r-Indice(curve_1)))

plt.subplot(1,2,2)
plt.plot(np.real(curve_2), np.imag(curve_2), linewidth=3, label='$\Delta(\mathbb{S})$')
plt.plot([np.min(np.real(curve_2)), np.max(np.real(curve_2))], [0,0], 'k')
plt.plot([0,0], [np.min(np.imag(curve_2)), np.max(np.imag(curve_2))], 'k')
plt.legend()
plt.title(u"$\u03bc_b$=10 // r-$Ind_{\Delta \mathbb{S}}=$"+str(schema_1.r-Indice(curve_2)))
plt.xlabel('Re')
plt.show()

###########################################
### Example 2.1: Figure 2.2 / 2.3 / 2.4 ###
###########################################

''' Imposed spectral properties '''
xi_K    = np.array([0, np.pi])
F_K     = np.array([1,1])
C_K = np.array([1,-1])

Schema  = FiniteDifferenceScheme(xi_K, F_K, C_K)
Schema.schemeGenerator()
Schema.isCauchyStable()

# Draw Symbol - Figure 2.2 #
Xi           = np.linspace(-np.pi, np.pi, 10**3)
spectre      = Schema.symbol(Xi)
cercle_unite = np.exp(1j*Xi)

plt.plot(np.real(spectre), np.imag(spectre), linewidth = 4, c='r')
plt.plot(np.real(cercle_unite), np.imag(cercle_unite), '--k', linewidth = 4)
plt.plot([-1,1],[0,0],'k')
plt.plot([0,0], [-1,1],'k')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.xlabel('Re')
plt.ylabel('Im')
plt.axis('equal')

plt.show()

# Simulation - Figure 2.3 #
J = 10**3      
N = 10*10**3

bound_l = Bd.Dirichlet()

bound_r = Bd.Neumann(kb = 1)

M_DF = Schema.matrixFiniteDifference(J, bound_r, bound_l) # Compute the Finite Difference matrix
                                                                   # associated to the scheme with dirichlet
                                                                   # boundary condition on the left boundary
                                                                   # and Neumann order kb on the right
                                                                   # boundary
U     = np.zeros((J+1,N), dtype='cfloat')  # save the solution for all time steps

def u_0(x):
    return np.exp(-50*(x-1/2)**2)*1 # np.exp(-50*(x-1/2)**2)*cos(np.pi*J*(x-1/2))


U[:,0]     = [u_0(j/(J+1)) for j in range(J+1)]   # Initial condition

for n in tqdm(range(1,N)):                  # Run the simulation 
    U[:, n] = M_DF.dot(U[:,n-1])

n = 2000
plt.figure(2)
plt.pcolormesh(np.linspace(0,1,J+2), np.linspace(0,n/(J+1),n+1), np.transpose(np.abs(U[:,:n])), cmap='hot')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('Time (t)')
plt.title('$U^n_j$')
plt.show()


# Simulation - Figure 2.4 #
U     = np.zeros((J+1,N), dtype='cfloat')  # save the solution for all time steps

U[J//2,0]     = 10 # Simulate a Dirach mass initial condition which triggers all the mods 

for n in tqdm(range(1,N)):                  # Run the simulation 
    U[:, n] = M_DF.dot(U[:,n-1])

Xt = np.arange(0,N)

n = 2000
plt.figure(2)
plt.pcolormesh(np.linspace(0,1,J+2), np.linspace(0,n/(J+1),n+1), np.transpose(np.abs(U[:,:n])), cmap='hot', vmax=0.5)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('Time (t)')
plt.title('$U^n_j$')
plt.show()


#####################################
### Example 2.2: Figure 2.5 / 2.6 ###
#####################################

''' Imposed spectral properties '''
xi_K    = np.array([0, np.pi, 0.5*np.pi])
F_K     = np.array([1,1,1])
C_K = np.array([1,-1,2])

Schema  = FiniteDifferenceScheme(xi_K, F_K, C_K)
Schema.schemeGenerator()
Schema.isCauchyStable()


# Draw Symbole - Figure 2.5 #
Xi           = np.linspace(-np.pi, np.pi, 10**3)
spectre      = Schema.symbol(Xi)
cercle_unite = np.exp(1j*Xi)

plt.plot(np.real(spectre), np.imag(spectre), linewidth = 4, c='r')
plt.plot(np.real(cercle_unite), np.imag(cercle_unite), '--k', linewidth = 4)
plt.plot([-1,1],[0,0],'k')
plt.plot([0,0], [-1,1],'k')
plt.axis([-1.5,1.5,-1.5,1.5])
plt.xlabel('Re')
plt.ylabel('Im')
plt.axis('equal')
plt.show()

# Simulation - Figure 2.6 #
J = 10**3
N = 10*10**3

bound_l = Bd.Dirichlet()

bound_r = Bd.Neumann(kb = 2)

M_DF = Schema.matrixFiniteDifference(J, bound_r, bound_l)        # Compute the Finite Difference matrix
                                                                   # associated to the scheme with dirichlet
                                                                   # boundary condition on the left boundary
                                                                   # and Neumann order kb on the right
                                                                   # boundary
U     = np.zeros((J+1,N), dtype='cfloat')  # save the solution for all time steps

RootsForward = Schema._Kappa(z0 = 1)[0]             # Compute all the forward roots, useful to define the IC

def u_0(x):
    ''' Define a initial condition which triggers only the forward mods
    which have modulus equal to 1, with an amplitude equal to 1 ''' 
    eps    = 10**(-10)                              # Treshold 
    result = 0
        
    for j in range(Schema.r):
        if abs(np.abs(RootsForward[j])-1)<eps:      # Keep only the one with modulus equal to 1  
            # Case the root = 1 or -1
            if abs(np.real(RootsForward[j])-1)<eps or abs(np.real(RootsForward[j])+1)<eps: 
                result += 1*RootsForward[j]**(J*(x-1/2))
            
            # Case the root is the half complex plan Imag > 0
            elif np.imag(RootsForward[j])>0:        
                result += -0.5*1j*RootsForward[j]**(J*(x-1/2))
            
            # Case the root is the half complex plan Imag < 0
            else:  
                result += 0.5*1j*RootsForward[j]**(J*(x-1/2))
    
    return (result+(-1)**(J*(x-1/2)))*np.exp(-50*(x-1/2)**2)    # Get a compactly supported IC


U[:,0]     = [u_0(j/(J+1)) for j in range(J+1)]   # Simulate a dirach initial condition which triggers all the mods 

for n in tqdm(range(1,N)):                  # Run the simulation 
    U[:, n] = M_DF.dot(U[:,n-1])

n = 2000
plt.figure(2)
plt.pcolormesh(np.linspace(0,1,J+2), np.linspace(0,n/(J+1),n+1), np.transpose(np.abs(U[:,:n])), cmap='hot', vmax=1)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('Time (t)')
plt.title('$U^n_j$')
plt.show()


####################################################################
### Subsection "Neumann ordre 2 ": Figure 2.7 / 2.8 / 2.9 / 2.10 ###
####################################################################

''' Imposed spectral properties '''
xi_K    = np.array([0, 0.3*np.pi, 0.8*np.pi])
F_K     = np.array([1,1, 1])
C_K = np.array([1,-1, 1])

Schema  = FiniteDifferenceScheme(xi_K, F_K, C_K)
Schema.schemeGenerator()

# Roots & Symbol - Figure 2.7 #
z0 = 1
[RootsFromInside, RootsFromOutside] = Schema._Kappa(z0)

plt.rcParams.update({'font.size': 30})
theta = np.linspace(0,2*np.pi,10**3)


plt.subplot(1,2,1)
plt.plot(np.cos(theta), np.sin(theta), 'k', linewidth = 5)
plt.scatter(np.real(RootsFromInside), np.imag(RootsFromInside), s=4**5, c='b', label='rightgoing')
plt.scatter(np.real(RootsFromOutside), np.imag(RootsFromOutside), s=4**5, c='r', label='leftgoing')
plt.legend()
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Racine de P(X) pour z0={}'.format(z0))

Xi           = np.linspace(-np.pi, np.pi, 10**3)
spectre      = Schema.symbol(Xi)
cercle_unite = np.exp(1j*Xi)

plt.subplot(1,2,2)
plt.plot(np.real(spectre), np.imag(spectre), linewidth = 4, c='r')
plt.plot(np.real(cercle_unite), np.imag(cercle_unite), '--k', linewidth = 4)
plt.plot([-1,1],[0,0],'k')
plt.plot([0,0], [-1,1],'k')
plt.axis([-1.1,1.1,-1.1,1.1])
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Symbole')

plt.show()

# Simulation - Figure 2.8 #
nbr_global_reflexion = 10
kb = 2

''' Condition initiale '''
def u_0(x):
    return np.exp(-50*(x-1/2)**2)*np.sin(0.8*np.pi*(x-1/2)*J) 

bound_l = Bd.Dirichlet()

bound_r = Bd.Neumann(kb = 2)

J = 10**3
N = 2000*nbr_global_reflexion

X = np.linspace(-0,1,J+1)
M_DF = Schema.matrixFiniteDifference(J, bound_r, bound_l)

U        = np.zeros((J+1,N))
U_l2     = np.zeros((N,1))
U[:,0]   = [u_0(x) for x in X]
U_l2[0,0]   = np.sqrt(np.sum(np.abs(U[:,0])**2)/(J+1))

for n in tqdm(range(1,N)):
    U[:, n]    = M_DF.dot(U[:,n-1])
    U_l2[n,0]   = np.sqrt(np.sum(np.abs(U[:,n])**2)/(J+1))

plt.rcParams.update({'font.size': 20})

Xt = np.arange(0,N)
plt.semilogy(Xt, U_l2, linewidth = 4)
plt.semilogy(Xt[1500:-1:2000], U_l2[1500:-1:2000], '-r', linewidth = 4)
for j in range(2*nbr_global_reflexion):
      plt.semilogy([500+j*1000, 500+j*1000], [0, np.max(U_l2)], '--k', linewidth = 2)
plt.title('$||U^n||_{l^2}$'+'pour Dirichlet/Neumann ordre {}'.format(kb))
plt.xlabel('n')
plt.grid()
plt.ticklabel_format(axis = 'x', style = 'scientific')
plt.show()

# Chock diagram - Figure 2.9 #
n = 7000

plt.pcolormesh(np.linspace(0,1,J+2), np.linspace(0,n/(J+1),n+1), np.abs(np.transpose(U[:,:n])), cmap='hot')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('$U^n_j$')
plt.show()

# Simulation - Figure 2.10 #
nbr_global_reflexion = 100
kb = 2

''' Condition initiale '''
def u_0(x):
    return np.exp(-50*(x-1/2)**2)*np.sin(0.8*np.pi*(x-1/2)*J) 

bound_l = Bd.Dirichlet()

bound_r = Bd.Neumann(kb = 2)

J = 10**3
N = 2000*nbr_global_reflexion

X = np.linspace(-0,1,J+1)
M_DF = Schema.matrixFiniteDifference(J, bound_r, bound_l)

U        = np.zeros((J+1,N))
U_l2     = np.zeros((N,1))
U[:,0]   = [u_0(x) for x in X]
U_l2[0,0]   = np.sqrt(np.sum(np.abs(U[:,0])**2)/(J+1))

for n in tqdm(range(1,N)):
    U[:, n]    = M_DF.dot(U[:,n-1])
    U_l2[n,0]   = np.sqrt(np.sum(np.abs(U[:,n])**2)/(J+1))

Xt = np.arange(0,N)
plt.semilogy(Xt, U_l2, linewidth = 4)
plt.semilogy(Xt[1500:-1:2000], U_l2[1500:-1:2000], '-r', linewidth = 4)
plt.title('$||U^n||_{l^2}$'+'pour Dirichlet/Neumann ordre {}'.format(kb))
plt.xlabel('n')
plt.grid()
plt.ticklabel_format(axis = 'x', style = 'scientific')
plt.show()

Schema.isCauchyStable()
pol = np.polyfit(Xt[1500:-1:2000], np.log(U_l2[1500:-1:2000,0]), 1)
print(' Coeff directeur pour abscisse n = ',np.exp(pol[0]))
print(' Coeff directeur pour abscisse double redonb = ',np.exp((np.log(U_l2[99*2000,0])-np.log(U_l2[50*2000,0]))/49))
print(' max eigvals(M) = ', np.max(np.abs(np.linalg.eigvals(M_DF))))
print('max eigvals(R) = ', np.max(np.abs(np.linalg.eigvals(Schema.matrixReflexion(1, bound_r, bound_l, J)))))



#########################################################
### Subsection "Neumann ordre 1 ": Figure 2.11 / 2.12 ###
#########################################################
''' Propriétés spectrales '''
xi_K    = np.array([0, 0.288*np.pi, 0.82*np.pi])
F_K     = np.array([1,1, 1])
C_K = np.array([1,-1, 1])

Schema  = FiniteDifferenceScheme(xi_K, F_K, C_K)
Schema.schemeGenerator()

# Roots & Symbol - Figure 2.11 #
z0 = 1
[RootsFromInside, RootsFromOutside] = Schema._Kappa(z0)

plt.rcParams.update({'font.size': 30})
theta = np.linspace(0,2*np.pi,10**3)


plt.subplot(1,2,1)
plt.plot(np.cos(theta), np.sin(theta), 'k', linewidth = 5)
plt.scatter(np.real(RootsFromInside), np.imag(RootsFromInside), s=4**5, c='b', label='rightgoing')
plt.scatter(np.real(RootsFromOutside), np.imag(RootsFromOutside), s=4**5, c='r', label='leftgoing')
plt.legend()
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Racine de P(X) pour z0={}'.format(z0))

Xi           = np.linspace(-np.pi, np.pi, 10**3)
spectre      = Schema.symbol(Xi)
cercle_unite = np.exp(1j*Xi)

plt.subplot(1,2,2)
plt.plot(np.real(spectre), np.imag(spectre), linewidth = 4, c='r')
plt.plot(np.real(cercle_unite), np.imag(cercle_unite), '--k', linewidth = 4)
plt.plot([-1,1],[0,0],'k')
plt.plot([0,0], [-1,1],'k')
plt.axis([-1.1,1.1,-1.1,1.1])
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Symbole')

plt.show()

# Simulation - Figure 2.12 #
nbr_double_rebond = 100

''' Condition initiale '''
def u_0(x):
    return np.exp(-50*(x-1/2)**2)*np.sin(0.82*np.pi*(x-1/2)*J) 

bound_l = Bd.Dirichlet()

bound_r = Bd.Neumann(kb = 1)

J = 994
N = 2000*nbr_double_rebond

X = np.linspace(-0,1,J+1)
M_DF = Schema.matrixFiniteDifference(J, bound_r, bound_l)

U_l2      = np.zeros((N,1))              
U        = np.zeros((J+1,N))
U[:,0]   = [u_0(x) for x in X]
U_l2[0,0]   = np.sqrt(np.sum(np.abs(U[:,0])**2)/(J+1))

for n in tqdm(range(1,N)):
    U[:, n]    = M_DF.dot(U[:,n-1])
    U_l2[n,0]   = np.sqrt(np.sum(np.abs(U[:,n])**2)/(J+1))
    
Xt = np.arange(0,N)

plt.rcParams.update({'font.size': 30})

plt.semilogy(Xt, U_l2, linewidth = 4)
plt.semilogy(Xt[1500:-1:2000], U_l2[1500:-1:2000], '--r', linewidth = 4)
plt.title('$||U^n||_{l^2}$'+'pour Dirichlet/Neumann ordre {}'.format(kb))
plt.xlabel('n')
plt.grid()
plt.show()

# chock diagram #
n = 7000

plt.pcolormesh(np.linspace(0,1,J+2), np.linspace(0,n/(J+1),n+1), np.abs(np.transpose(U[:,:n])), cmap='hot')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('$U^n_j$')
plt.show()

Schema.isCauchyStable()
pol = np.polyfit(Xt[1500:-1:2000], np.log(U_l2[1500:-1:2000,0]), 1)
print(' Coeff directeur pour abscisse n = ',np.exp(pol[0]))
print(' Coeff directeur pour abscisse double redonb = ',np.exp((np.log(U_l2[99*2000,0])-np.log(U_l2[80*2000,0]))/19))
print(' max eigvals(M) = ', np.max(np.abs(np.linalg.eigvals(M_DF))))
print('max eigvals(R) = ', np.max(np.abs(np.linalg.eigvals(Schema.matrixReflexion(1, bound_r, bound_l, J)))))


##################################################
### Conter example : Figure 2.13 / 2.14 / 2.15 ###
##################################################
xi_K    = np.array([0, 0.2*np.pi, np.pi, 0.7*np.pi, 0.5*np.pi])
F_K     = np.array([1,1, 1,1,1])
C_K = np.array([1,-1,1.3,0.1, -0.8])

Schema  = FiniteDifferenceScheme(xi_K, F_K, C_K)
Schema.schemeGenerator()
Schema.isCauchyStable()

# Roots & Symbol - Figure 2.13 #
z0 = 1
[RootsFromInside, RootsFromOutside] = Schema._Kappa(z0)

plt.rcParams.update({'font.size': 30})
theta = np.linspace(0,2*np.pi,10**3)


plt.subplot(1,2,1)
plt.plot(np.cos(theta), np.sin(theta), 'k', linewidth = 5)
plt.scatter(np.real(RootsFromInside), np.imag(RootsFromInside), s=4**5, c='b', label='rightgoing')
plt.scatter(np.real(RootsFromOutside), np.imag(RootsFromOutside), s=4**5, c='r', label='leftgoing')
plt.legend()
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Racine de P(X) pour z0={}'.format(z0))

Xi           = np.linspace(-np.pi, np.pi, 10**3)
spectre      = Schema.symbol(Xi)
cercle_unite = np.exp(1j*Xi)

plt.subplot(1,2,2)
plt.plot(np.real(spectre), np.imag(spectre), linewidth = 4, c='r')
plt.plot(np.real(cercle_unite), np.imag(cercle_unite), '--k', linewidth = 4)
plt.plot([-1,1],[0,0],'k')
plt.plot([0,0], [-1,1],'k')
plt.axis([-1.1,1.1,-1.1,1.1])
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Symbole')

plt.show()

# Simulation - Figure 2.14 / 2.15 #

J = 987 #981

N = 2000*100

X = np.linspace(-0,1,J+1)

def u_0(x):
    return np.exp(-50*(x-1/2)**2)*(np.sin(0.2*np.pi*(x-1/2)*J)+np.cos(np.pi*(x-1/2)*J)) 

bound_l = Bd.Dirichlet()

bound_r = Bd.Neumann(kb = 2)

M_DF = Schema.matrixFiniteDifference(J, bound_r, bould_l)

U_l2      = np.zeros((N,1))              
U        = np.zeros((J+1,N))

U[:,0]   = [u_0(x) for x in X]
U_l2[0,0]   = np.sqrt(np.sum(np.abs(U[:,0])**2)/(J+1))

for n in tqdm(range(1,N)):
    U[:, n]    = M.dot(U[:,n-1])
    U_l2[n,0]   = np.sqrt(np.sum(np.abs(U[:,n])**2)/(J+1))
    
Xt = np.arange(0,N)

plt.rcParams.update({'font.size': 30})

plt.semilogy(Xt, U_l2, linewidth = 4)
plt.title('$||U^n||_{l^2}$'+'pour Dirichlet/Neumann ordre {}'.format(kb))
plt.xlabel('n')
plt.grid()
plt.show()

print('rayon spectral (R) = ', np.max(np.abs(np.linalg.eigvals(Schema.matrice_reflexion(1, bound_r, bound_l,J)))))  
print('rayon spectral (M) = ', np.max(np.abs(np.linalg.eigvals(M_DF))))  
