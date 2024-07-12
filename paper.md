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

In condensed matter physics, a standard methodology for modeling materials is the *tight-binding model*. In the context of electronic systems (e.g. metals), the electrons in such a model typically "live" at one atomic site, but from time to time "hop" over to neighboring atoms. By including a spin structure as well in this formalism – meaning that we keep track of what spins each electron has, and whether the spins "flip" during various interactions that are permitted on this lattice – we can model a wide variety of physical phenomena including superconductivity and magnetism. Mathematically, this is often expressed in the language of quantum field theory: We define one operator $c^\dagger_{i\sigma}$ that "puts" an electron with spin $\sigma \in \{\uparrow, \downarrow\}$ on an atomic site with some index $i$, and another operator $c_{i\sigma}$ that "removes" a corresponding electron. The Hamiltonian operator $\mathcal{H}$ of the system is then constructed out of these electron operators – and this can in turn be used to calculate e.g. the ground-state energy, electric currents, superconducting order parameters, and properties of interest.

To do anything useful with that Hamiltonian *on a computer*, however, you typically have to translate it to a matrix form. This is where Bodge enters the picture:

- It provides an easy-to-use Pythonic interface for constructing the Hamiltonian of a tight-binding system. Particular focus has been placed on making it easy to describe systems that include various forms of superconductivity and magnetism, making it a great choice for modeling e.g. superconductivity in magnetic heterostructures.
- It focuses on high performance. For efficiency, it uses SciPy sparse matrices internally [@scipy_2020], and it manages to construct large Hamiltonians in $\mathcal{O}(N)$ time where $N$ is the number of sites. According to my benchmarks (see the [documentation](https://jabirali.github.io/bodge/)), the performance is similar to [Kwant](https://kwant-project.org/) [@groth_kwant_2014], which is the state of the art for numerical condensed matter physics. The results can be returned in most useful NumPy or SciPy matrix formats.
- It is designed to be extensible. For instance, Bodge is designed to support any lattice type – and you can implement your own by subclassing `Lattice` and implementing 2-3 short iterators for your new lattice.
- Some convenience methods are provided to help you with the next steps of your calculations: Extracting the local density of states (LDOS), calculating the free energy, diagonalizing the Hamiltonian, etc. (Some more advanced algorithms live on the `development` branch, but have not yet been assimilated into the `main` branch.)
- The code itself follows modern software development practices: High test coverage with continuous integration (via `pytest`), fast runtime type checking (via `beartype`), and high PEP-8 compliance (via `black`).

There are two main alternatives that arguably fill the same niche as Bodge: Kwant [@groth_kwant_2014] and Pybinding [@moldovan_pybinding_2020].

# Examples

Introductory examples of how to use Bodge are provided in the [official documentation](https://jabirali.github.io/bodge/). Examples of research problems that have been studied using Bodge include superconductor/altermagnet heterostructures [@ouassou_alt_2023] and RKKY interactions in unconventional superconductors [@ouassou_rkky_2024; @ouassou_dmi_2024].

# Acknowledgements

I acknowledge very helpful discussions with my PostDoc supervisor Prof. Jacob Linder when learning the BdG formalism as well as on superconductivity in general. I also acknowledge some useful discussions with Morten Amundsen, Henning G. Hugdal, and Sol H. Jacobsen on tight-binding modeling in general.

This work was supported by the Research Council of Norway through Grant No. 323766 and its Centres of Excellence funding scheme Grant No. 262633 "QuSpin." During the development of this package, some numerical calculations were performed on resources provided by Sigma2 – the National Infrastructure for High Performance Computing and Data Storage in Norway, Project No. NN9577K. The work presented in this paper has also benefited from the Experimental Infrastructure for Exploration of Exascale Computing (eX3), which is financially supported by the Research Council of Norway under contract 270053.

# References
