---
title: "Bodge documentation"
date: 2024-07-09
author:
  - name: Jabir Ali Ouassou
    email: jabir.ali.ouassou@hvl.no
    url: https://scholar.google.com/citations?user=SbyugkkAAAAJ
abstract: > 
  Bodge is a Python package for constructing large real-space tight-binding
  models. Although quite general tight-binding models can be constructed, we
  focus on the Bogoliubov-DeGennes ("BoDGe") Hamiltonian, which is used to
  model superconductivity in clean materials. So if you want a lattice model
  for superconducting nanostructures, and want something that is efficient yet
  easy to use, then you've come to the right place.
---


# Installation

Bodge has been [uploaded](https://pypi.org/project/bodge/) to the official PyPi package repository. This means that if you have a recent version of Python and Pip installed on your system, installing this package should be as simple as:

    pip install bodge

For more installation alternatives, please see the [README file on GitHub](https://github.com/jabirali/bodge).

# Mathematical background

In condensed matter physics, one usually describes the Hamiltonian $\mathcal{H}$ of a system using the quantum field theory. In the context of crystal lattices, the building blocks are then one operator $c_{i\sigma}^\dagger$ that "puts" an electron with spin $\sigma \in \{\uparrow, \downarrow\}$ at a lattice site described by some index $i$, and another operator $c_{i\sigma}$ that "removes" a corresponding electron from that site. One can then describe many physical phenomena in this language: $c^\dagger_{i+1,\uparrow} c_{i,\downarrow}$ would e.g. remove a spin-down electron from site $i$ and place a spin-up electron at site $i+1$, thus describing an electron that "hops" between two lattice sites while flipping its spin. By summing up the different allowed processes on the lattice, the Hamiltonian operator $\mathcal{H}$ contains a description of all the permitted processes in our model – which can then be used to determine the system's ground state, order parameters, electric currents, etc.

We here focus on systems that can harbor superconductivity, which is often modeled using variants of the "Bogoliubov-deGennes Hamiltonian". In a very general form, such a Hamiltonian operator can be written:
$$\mathcal{H} = E_0 + \frac{1}{2} \sum_{ij} \hat{c}^\dagger_i \hat{H}_{ij} \hat{c}_j,$$
where $\hat{c}_i = (c_{i\uparrow}, c_{i\downarrow}, c_{i\uparrow}^\dagger, c_{i\downarrow}^\dagger)$ is a vector of all spin-dependent electron operators on lattice site $i$ and $E_0$ is a constant. The $4\times4$ matrix $\hat{H}_{ij}$ is generally further decomposed into $2\times2$ blocks $H_{ij}$ and $\Delta_{ij}$:
$$\hat{H}_{ij} = \begin{pmatrix} H_{ij} & \Delta_{ij} \\ \Delta^\dagger_{ij} & -H^*_{ij} \end{pmatrix}.$$
Physically, the matrix $H_{ij}$ describes all the non-superconducting components in the system. A typical example of a non-magnetic system – and what some people might call *the* tight-binding model – would be:
$$H_{ij} = \begin{cases} -\mu\sigma_0 & \text{if $i = j$,} \\ -t\sigma_0 & \text{if $i, j$ are neighbors,} \\ 0 & \text{otherwise.} \end{cases}$$
Here, $\sigma_0$ is a $2\times2$ identity matrix, signifying that the Hamiltonian has no spin structure and therefore no magnetic properties.
The constant $\mu$ is the chemical potential and provides a contribution to the Hamiltonian for every present electron, while the constant $t$ is the hopping amplitude which parametrizes how mobile the electrons are. In magnetic systems, one can include the Pauli vector $\boldsymbol{\sigma} = (\sigma_1, \sigma_2, \sigma_3)$ in on-site terms (first row) to model ferromagnets and antiferromagnets, or in nearest-neighbor terms (second row) to model altermagnets and spin-orbit coupling.

The other matrix $\Delta_{ij}$ represents electron-electron pairing, and models different forms of superconductivity on the lattice. The simplest version is the conventional Bardeen–Cooper–Schrieffer (BCS) superconductivity, also known as "$s$-wave spin-singlet superconductivity". This is most easily modeled using an on-site electron-electron pairing:
$$\Delta_{ij} = \begin{cases} -\Delta_s i\sigma_2 & \text{if $i = j$,} \\ 0 & \text{otherwise.} \end{cases}$$
But the same formalism can be used to model other types of "unconventional" superconductivity. For instance, the $d$-wave superconductivity that appears at "high temperatures" (liquid nitrogen) in cuprates can be described by the more complicated expression
$$\Delta_{ij} = \begin{cases} -\Delta_d i\sigma_2 & \text{if $i$ and $j$ are neighbors along the $x$ axis,} \\ +\Delta_d i\sigma_2 & \text{if $i$ and $j$ are neighbors along the $y$ axis,} \\ 0 & \text{otherwise.} \end{cases}$$

Thus, we have motivated that the equation above is able to describe a quite general class of materials. Bodge essentially provides an interface that lets you directly set the elements of $H_{ij}$ and $\Delta_{ij}$ using a Python context manager, thus simplifying the process of constructing tight-binding models on a computer. The main assumption behind the Bodge implementation is that we are *only* interested in on-site and nearest-neighbor interactions, and not e.g. next-nearest-neighbor interactions (sometimes used to study magnetic frustration) or long-distance interactions (sometimes relevant for strongly correlated systems). For the majority of tight-binding models, nearest-neighbor interactions is however an acceptable and realistic cutoff, and this assumption makes the Bodge source code faster and cleaner.

# Getting started

This section is still under construction.

# Examples

This section is still under construction.
