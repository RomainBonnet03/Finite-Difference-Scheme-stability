# Finite-Difference-Scheme-Stability
## Motivation of the project
This project is the result of a Master Thesis, carried out at the 'Institut des Mathématiques de Toulouse' (Institute of Mathematics of Toulouse) under the supervision of Jean-François Coulombel and Grégory Faye, on the subject of "Asymptotic Analysis and Stability of Finite Difference Schemes in bounded domain".

### Introduction
The $l^2$ stability of Finite Difference (FD) schemes defined over $\mathbb{Z}$ is described by the Von-Neumann analysis. For domains bounded on one side, for exemple for schemes defined over $\mathbb{N}$, the GKS theory (Gustafsson, Kreiss, Sundström) is the most general approach (for hyperbolic problems) despite being often described as complicated. Trefethen offers a geometrical and physical interpretation of the GKS criteria which is based on a modal analysis. Let consider a domain bouded on the left side, and let consider a solution of the scheme under the form $U^n_j = z^n\kappa^j$ with $z,~\kappa\in\mathbb{C}$ with $|z|>1$. Trefethen interprets the mods $|\kappa|<1$ as "rightgoing" mods and the mods $|\kappa|>1$ as "leftgoing" mods. **If the rightgoing mods can emerge from the left boundary without being excited by leftgoing mods, then it is unstable**. The sensitive issue occures as $z\rightarrow z_0$ with $|z_0|=1$. Trefethen states a numerical scheme is a dispersive system which allows the propagation of wave packets characterized by their wave number $\xi$, frequency $\omega$ and group velocity $C$ ($\xi$ and $\omega$ are linked through the dispersion relation). Then $z^n=e^{i\omega n}$ and $\kappa^j=e^{-i\xi j}$. He demonstrates that the mods with a negative group velocity are the limit of leftgoing mods whereas the ones with positive group velocity are the limit of rightgoing modes. Thus, he strengthened the previous necessary condition to obtain a necessary and sufficient condition: **if mods with positive group velocity can propagates without being triggered by a mod with negative group velocity, then the scheme is GKS unstable**.  

Adressing the question of stability on a interval, Trefethen proves that the combination of two GKS-stable boundaries does not imply the overall stability of the model. One can indeed imagine a wave packet being trapped and reflected by the two interfaces and if the total reflection is greater than 1, then the amplitude of the solution grows exponentially with time: we say that the scheme is semi-group unstable. 

To learn more about the theoretical background, please refer to the manuscript.

### Problematic adressed
This Master Thesis exhibits examples of exponentially-unstable FD scheme, consistant with the transport equation, defined on an interval with a Dirichlet boundary condition on the incoming edge (physical boundary condition) and a Neumann boundary condition on the outgoing edge (transparent boundary condition).

## Description of the project
This project is composed of two main classes: *FiniteDifferenceScheme* and *Boundaries*.

- The class *Boundaries*: is an object which encodes the properties of the boundary condition which are:
  - $m$ the number of nodes (j=0, ..., m) needed to compute the value of the solution at the ghost points
  - $B$ a matrix which encodes the coefficient $b_{k,j}$ of the boundary condition  
- The class *FiniteDifferenceScheme*: is an object which encodes the properties of the studied Finite Difference scheme. It takes as inputs the spectral properties (the wave numbers $\xi_k$, the associated temporal mods $z_k$ and the group velocities $C_k$) the scheme must satisfy. The method *schemeGenerator()* then solves the system to compute the coefficients and the stencil of the scheme which verifies the prescribed spectral properties. The class also includes the following methods:
  - *symbol(self, \theta)* which computes the value of the symbol, denoted $F$, of the scheme for a given $\theta$. The method *drawSymbol(self)* plots the curve $F([0,~2\pi])$. The method *isCauchyStable(self)* verifies if the scheme if Cauchy-stable.
  - *drawRoots(self, z)* which plots the roots of the characteristic polynomial associated to the scheme for the temporal mod $z$. It is based on methods introduced by LeBarbenchon's work.
  - *matrixReflexion(self, z, bound-r, bound-l, J)* which compute the reflexion matrix of the scheme combined with bound-r (*Boundaries* object) boundary condition on the right edge and  bound-l (*Boundaries* object) boundary condition on the left edge, for a given temporal mod $z$ and a discretization of $J+1$ nodes.
  - *isSemiGroupStable(self, z, bound-r, bound-l, J)* which verify if the scheme scheme combined with bound-r (*Boundaries* object) boundary condition on the right edge and  bound-l (*Boundaries* object) boundary condition on the left edge, for a given temporal mod $z$ and a discretization of $J+1$ nodes, is semi-group stable. It is based on the criteria of Proposition 2.9 (cf manuscript).
  - *matrixFiniteDifference(self, J, bound-r, bound-l)* which computes the FD matrix (for bound-r as boundary condition on the right edge and bound-l on the left edge), denoted $M$ which allows to compute the solution of the scheme $U^n_j$. 

This project also relies on codes produced by Pierre LeBarbenchon for his PhD thesis which can be found: https://github.com/PLeBarbenchon/boundaryscheme. The corresponding license can be found in the repository.

**A more detailed description of the code can be found in the manuscript**.

## Example
Let first import the libraries and packages:
```
# Python libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Personal libraries
from FiniteDifferenceScheme import *
from Boundaries import *
```
Let compute our first finite difference scheme by imposing spectral properties: we want the scheme to admit $\kappa = 1$ and $\kappa = \pi$ as spatial mods, associated to the temporal mod $z=1$ and group velocities $1$ and $-1$ (respectively).
```
xi_K    = np.array([0, np.pi])
z_K     = np.array([1,1])                              # All the spatial modes are
                                                       # associated to z = 1
C_K = np.array([1,-1])
```
Let generate the corresponding scheme.
```
scheme  = FiniteDifferenceScheme(xi_K, z_K, C_K)      # Initialize the objet
scheme.schemeGenerator() 
```
One can verify that the scheme indeed satisfied the prescribed spectral properties:
```
scheme.drawRoots(z=1) 
```
Let now define the two boundary conditions: homogenous Dirichlet condition for the left edge (incoming edge) and homogenous Neumann of order $k_b=1$ condition for the right edge (outgoing edge):
```
bound_r  = Neumann(kb=2)

bound_l  = Dirichlet()
```
We can compute the FD matrix and simulate the system over a specified number of time steps:
```
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
```
It appears that the scheme we defined (completed with the specified boundary conditions) is semi-group unstable. We can confirm it with the computation of the reflexion matrix:
```
scheme.isSemiGroupStable(z=1, bound_r, bound_l, J)
```
which prints 'True' as result. Furthermore:
```
R = scheme.matrixReflexion(z=1, bound_r, bound_l, J)
spectral_radius = np.max(np.abs(np.linalg.eigvals(R)))    # Compute the spectral radius of the reflexion matrix
print(spectral_radius)
```
which prints $\rho(R_J) \approx <1$ (cf Proposition 2.9 of the manuscript)
