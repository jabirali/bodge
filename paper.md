---
title: 'Bodge: Python package for efficient tight-binding modeling of superconducting nanostructures'
tags:
  - python
  - numerical physics
  - condensed matter physics
  - tight-binding models
  - superconductivity
  - sparse matrices
  - bdg equations
authors:
  - name: Jabir Ali Ouassou
    orcid: 0000-0002-3725-0885
    email: jabir.ali.ouassou@hvl.no
    affiliation: "1, 2"
affiliations:
  - name: Department of Computer Science, Electrical Engineering and Mathematical Sciences, Western Norway University of Applied Sciences, NO-5528 Haugesund, Norway
    index: 1
  - name: Center for Quantum Spintronics, Department of Physics, Norwegian University of Science and Technology, NO-7491 Trondheim, Norway
    index: 2

date: 12 July 2024
bibliography: paper.bib
---

# Summary

[Bodge](https://github.com/jabirali/bodge) is Python package for constructing *large-scale real-space tight-binding models* for calculations in condensed matter physics. "Large-scale" means that it should remain performant even for lattices with millions of atoms, and "real-space" means that the model is formulated in terms of individual lattice sites and not in e.g. momentum space.

Although general tight-binding models can be constructed with this package, the main focus is on the Bogoliubov-DeGennes ("BoDGe") Hamiltonian used to model superconductivity in the clean limit [@zhu_bdg_2016; @degennes_bdg_1966]. The package is designed to be easy to use, flexible, and extensible – and very few lines of code are required to model heterostructures containing e.g. conventional and unconventional superconductors, ferromagnets and antiferromagnets, altermagnetism and spin-orbit coupling, etc.

In other words: If you want a lattice model for superconducting nanostructures, and want something that is computationally efficient yet easy to use, Bodge should be a good choice.

# Statement of need

In condensed matter physics, a standard methodology for modeling materials is the *tight-binding model*. In the context of electronic systems (e.g. metals), the electrons in such a model typically "live" at one atomic site, but from time to time "hop" over to neighboring atoms. By including a spin structure as well in this formalism – meaning that we keep track of what spins each electron has, and whether the spins "flip" during various interactions that are permitted on this lattice – we can model a wide variety of physical phenomena including superconductivity and magnetism. Mathematically, this is often expressed in the language of quantum field theory: We define one operator $c^\dagger_{i\sigma}$ that "puts" an electron with spin $\sigma \in \{\uparrow, \downarrow\}$ on an atomic site with some index $i$, and another operator $c_{i\sigma}$ that "removes" a corresponding electron. The Hamiltonian operator $\mathcal{H}$ of the system is then constructed out of these electron operators – and this can in turn be used to calculate e.g. the ground-state energy, electric currents, superconducting order parameters, and other relevant material properties.

To do anything useful with that Hamiltonian *on a computer*, however, you typically have to translate it to a matrix form. This is where Bodge enters the picture:

- It provides an easy-to-use Pythonic interface for constructing the Hamiltonian of a tight-binding system. Particular focus has been placed on making it easy to describe systems that include various forms of superconductivity and magnetism, making it a great choice for modeling e.g. superconductivity in magnetic heterostructures.
- It scales well to large systems. For efficiency, it uses SciPy sparse matrices internally [@scipy_2020], and it constructs large Hamiltonians in $\mathcal{O}(N)$ time and memory where $N$ is the number of sites. According to [my benchmarks](https://jabirali.github.io/bodge/tutorial.html#numerical-details), the performance is similar to [Kwant](https://kwant-project.org/) [@groth_kwant_2014], which is the state of the art for numerical condensed matter physics. The results can be returned in most NumPy or SciPy matrix formats.
- It is designed to be extensible. For instance, while Bodge currently only implements square and cubic lattices (via the `CubicLattice` class), it can be used to construct Hamiltonians on e.g. triangular or hexagonal lattices if you want: you just need to subclass `Lattice` and implement 2-3 short iterators that describe your lattice.
- Some convenience methods are provided to help you with the next steps of your calculations: Extracting the local density of states (LDOS), calculating the free energy, diagonalizing the Hamiltonian, etc. (Some more advanced algorithms live on the `development` branch, but have not yet been assimilated into the `main` branch.)
- The code itself follows modern software development practices: Full test coverage with continuous integration (via `pytest`), fast runtime type checking (via `beartype`), and PEP-8 compliance (via `black`).

There are two main alternatives that arguably fill a similar niche to Bodge: Kwant [@groth_kwant_2014] and Pybinding [@moldovan_pybinding_2020]. Compared to these packages, the main benefit of Bodge is the focus on the BdG Hamiltonian in particular. For instance, using Kwant, it is up to the user to declare that each lattice site has 4 degrees of freedom (spin-up electrons, spin-down electrons, spin-up holes, and spin-down holes), and to ensure that you construct a Hamiltonian with the correct particle-hole symmetries. Bodge, however, *assumes* that these are the only relevant degrees of freedom, and enforces the relevant symmetries by default. In practice, this means that Kwant can be used to study a broader variety of physical systems, whereas Bodge can provide a friendlier syntax for users who work specifically on superconducting systems. Both packages support both NumPy arrays and SciPy sparse matrices as output formats, and both provide similar performance in the limit of large systems.

*Sparse matrix formats*, such as e.g. the *Compressed Sparse Row* (CSR) format used in the example below, have the advantage that they basically only store the non-zero elements of a matrix. In the case of a typical tight-binding model with nearest-neighbor hopping terms, the Hamiltonian matrix of a system of $N$ atoms has $\mathcal{O}(N^2)$ elements where only $\mathcal{O}(N)$ are actually non-zero. Thus, algorithms that leverage sparse matrices often result in at least an $\mathcal{O}(N)$ numerical speed-up, which becomes highly significant for large systems. With additional approximations, even larger speed-ups can in some cases be possible [@nagai_krylov_2020], although it depends on your system size and the strength of its interactions whether such approximations provide a good trade-off between accuracy and speed. However, *dense matrices* (i.e. NumPy arrays) allow for simpler solution algorithms based on e.g. full matrix diagonalization, and can become faster than sparse matrix algorithms for small systems or when leveraging GPU calculations. For this reason, both sparse and dense matrices are fully supported by Bodge, allowing the user to pick the most suitable matrix format for the task at hand.

# Examples

Introductory examples of how to use Bodge are provided in the [official documentation](https://jabirali.github.io/bodge/). Examples of research problems that have been studied using Bodge include superconductor/altermagnet heterostructures [@ouassou_alt_2023] and RKKY interactions in unconventional superconductors [@ouassou_rkky_2024; @ouassou_dmi_2024]. These papers also describe some sparse matrix algorithms that have been used to speed up numerical calculations based on Bodge, including Chebyshev expansion of the Fermi matrix [@weisse_chebyshev_2006; @benfenati_fermi_2022; @goedecker_fermi_1994] and a quick way to calculate the local density of states [@nagai_ldos_2017].

For a simple example of how this package can be used, consider a $64a\times64a$ square lattice. Let's assume that the whole system is a metal with chemical potential $\mu = 1.5t$, where $t=1$ is the hopping amplitude. Moreover, let's assume that half the system ($x < 32a$) is a conventional $s$-wave superconductor with an order parameter $\Delta_s = 0.1t$, whereas the other half is a ferromagnet with exchange field $\mathbf{M} = M_z \mathbf{e}_z$. To construct the Hamiltonian matrix for such a system in the CSR sparse format, we can use the following code:
```python
from bodge import *

# Tight-binding parameters
t = 1
μ = 1.5 * t
Δs = 0.1 * t
Mz = 0.5 * Δs

# Construct the Hamiltonian
lattice = CubicLattice((64, 64, 1))
system = Hamiltonian(lattice)

with system as (H, Δ):
    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
    for i in lattice.sites():
        if i[0] < 32:
            H[i, i] = -μ * σ0
            Δ[i, i] = -Δs * jσ2
        else:
            H[i, i] = -μ * σ0 - Mz * σ3

H = system.matrix(format="csr")
```
Note the use of a context manager (`with`-block) to provide an intuitive array syntax for accessing the relevant parts of the Hamiltonian matrix, while abstracting away the underlying sparse matrix details. You can use this Hamiltonian to easily implement your own sparse matrix algorithms. Alternatively, Bodge provides several convenience methods to e.g. calculate the density of states, free energy, etc. More such methods are currently under development.

# Acknowledgements

I acknowledge very helpful discussions with my PostDoc supervisor Prof. Jacob Linder when learning the BdG formalism. I also acknowledge useful discussions with Morten Amundsen, Henning G. Hugdal, and Sol H. Jacobsen on tight-binding modeling in general. I also want to thank Mayeul d'Avezac and Yue-Wen Fang for providing helpful input during the referee process, which improved the Bodge software package.

This work was supported by the Research Council of Norway through Grant No. 323766 and its Centres of Excellence funding scheme Grant No. 262633 "QuSpin." During the development of this package, some numerical calculations were performed on resources provided by Sigma2 – the National Infrastructure for High Performance Computing and Data Storage in Norway, Project No. NN9577K. The work presented in this paper has also benefited from the Experimental Infrastructure for Exploration of Exascale Computing (eX3), which is financially supported by the Research Council of Norway under contract 270053.

# References
