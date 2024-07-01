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

- The class *Boundaries*: is an object which encodes the properties of the boundary condition.
- The class *FiniteDifferenceScheme*: is an object which encodes the properties of the studied Finite Difference scheme. It takes as inputs the spectral properties the scheme must satisfy. The method *schemeGenerator()* then solves the system to compute the coefficients and the stencil of the scheme which verifies the prescribed spectral properties. The class also includes methods which verify if the scheme is Cauchy-stable, if the scheme if semi-group stable for a given mod $|z|=1$, a given discretization of $J+1$ nodes and given boundary conditions (*Boundaries* objects).

This project also relies on codes produced by Pierre LeBarbenchon for his PhD thesis which can be found: https://github.com/PLeBarbenchon/boundaryscheme. The corresponding license 


## Example
