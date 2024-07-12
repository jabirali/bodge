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

[Bodge](https://github.com/jabirali/bodge) is Python package for constructing *large-scale real-space tight-binding models* for calculations in condensed matter physics. "Large-scale" means that it uses SciPy sparse matrices [@scipy_2020] to scale to lattices with millions of atoms, and "real-space" means that the model is formulated in terms of individual lattice sites and not in e.g. momentum space.

Although general tight-binding models can be constructed with this package, the main focus is on the Bogoliubov-DeGennes ("BoDGe") Hamiltonian which is used to model superconductivity in the clean limit [@zhu_bdg_2016; @degennes_bdg_1966]. The package is designed to be extensible and flexible, and can easily be used to model heterostructures containing e.g. conventional and unconventional superconductors, ferromagnets and antiferromagnets, altermagnetism and spin-orbit coupling, etc. In other words: If you want a lattice model for superconducting nanostructures, and want something that is computationally efficient yet easy to use, I believe that Bodge is a good choice.

# Statement of need

TODO

# Features

TODO

# Examples

Introductory examples of how to use Bodge are provided in the [official documentation](https://jabirali.github.io/bodge/). Examples of research problems that have been studied using Bodge include superconductor/altermagnet heterostructures [@ouassou_alt_2023] and RKKY interactions in unconventional superconductors [@ouassou_rkky_2024; @ouassou_dmi_2024].

# Acknowledgements

I acknowledge very helpful discussions with my PostDoc supervisor Prof. Jacob Linder when learning the BdG formalism as well as on superconductivity in general. I also acknowledge some useful discussions with Morten Amundsen, Henning G. Hugdal, and Sol H. Jacobsen on tight-binding modeling in general.

This work was supported by the Research Council of Norway through Grant No. 323766 and its Centres of Excellence funding scheme Grant No. 262633 "QuSpin." During the development of this package, some numerical calculations were performed on resources provided by Sigma2 – the National Infrastructure for High Performance Computing and Data Storage in Norway, Project No. NN9577K. The work presented in this paper has also benefited from the Experimental Infrastructure for Exploration of Exascale Computing (eX3), which is financially supported by the Research Council of Norway under contract 270053.

# References
