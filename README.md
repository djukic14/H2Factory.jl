# H2Factory

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="H2Factory logo" height="190">
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://djukic14.github.io/H2Factory.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://djukic14.github.io/H2Factory.jl/dev/)
[![Build Status](https://github.com/djukic14/H2Factory.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/djukic14/H2Factory.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/djukic14/H2Factory.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/djukic14/H2Factory.jl)

## Introduction

The H2Factory package provides Julia building blocks for hierarchical fast
matrix-vector products. It uses [H2Trees](https://github.com/djukic14/H2Trees.jl)
for trees and traversal plans, and supplies the multiplication routines and storage
used by a specific fast method.

The package provides:

* **Aggregation and disaggregation**: Tree passes that collect source moments and distribute the resulting fields.
* **Translations**: Interfaces for level transfers and far-field couplings, including dense, Toeplitz, and symmetry-reduced storage.
* **Moments**: Moment collections, pooled storage, and reusable buffers.
* **Near interactions**: Sparse, block, and user-supplied near-field operators.
* **Maps and workspaces**: A shared core for read-only method data and separate workspaces for mutable buffers.
* **Diagnostics**: Map summaries, timings, and checks for moment aliasing.

A fast method supplies its moment collections, translations, and near-field operator.
H2Factory combines them into a linear map with forward, transpose, and adjoint multiplication.

## Documentation

* Documentation for the [latest stable version](https://djukic14.github.io/H2Factory.jl/stable/).
* Documentation for the [development version](https://djukic14.github.io/H2Factory.jl/dev/).
* [Implementing a method](docs/src/implementing/required.md) and a [worked example](docs/src/implementing/example.md).
