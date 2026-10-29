# Flux Equilibrium Clustering

## Overview

This project implements **Flux Equilibrium Clustering (FEC)**, a density-seeking, graph-based clustering algorithm. The algorithm models each data point as a source of flux that flows through a weighted k-nearest-neighbor graph toward the global centroid. As flux redistributes over successive iterations, certain points become sinks that serve as cluster centers.

The implementation assigns each input point to a cluster based on its flow through the graph, providing an alternative to traditional clustering approaches such as k-means.

## Features

- **Graph construction:** Build a symmetric k-nearest-neighbor graph with Gaussian kernel edge weights.
- **Flux initialization and transport:** Simulate flux redistribution across graph edges over multiple iterations.
- **Flux-aware weighting:** Adjust edge weights based on accumulated flux.
- **Sink detection:** Identify vertices with sufficiently low outgoing flux to serve as cluster centers.
- **Cluster assignment:** Assign points to clusters by tracing steepest downhill paths toward sinks.
- **Cycle detection:** Detect cycles in paths and resolve them by designating additional sinks.
- **Small-cluster merging:** Merge clusters below a minimum size into nearby larger clusters.

## Algorithm

The implementation follows these main steps:

1. **Preprocessing:** Compute the global centroid, point-to-centroid distances, bandwidth parameter, and effective number of nearest neighbors.
2. **Graph construction:** Build a weighted k-nearest-neighbor graph using Gaussian kernel weights based on pairwise distances.
3. **Flux transport:** Initialize each point's flux and iteratively redistribute it among eligible downhill neighbors.
4. **Sink detection:** Identify points with outgoing flux below a specified threshold.
5. **Cluster assignment:** Trace each point's steepest downhill path to a sink, using cycle detection to handle loops.
6. **Post-processing:** Merge undersized clusters into nearby larger clusters.

## Parameters

The algorithm supports the following parameters, with defaults specified by the assignment:

| Parameter | Description | Default |
|---|---|---|
| `k` | Number of nearest neighbors | `10` |
| `sigma` | Gaussian kernel bandwidth | `0.2 ×` median pairwise distance |
| `alpha` | Fraction of stored flux redistributed per iteration | `0.5` |
| `T` | Number of transport iterations | `25` |
| `epsilon` | Threshold for sink detection | `1e-4` |
| `beta` | Flux reinforcement parameter | `None` |

## Technologies

- Python 3.10.14
- NumPy 2.2.6
- Pytest
- Python standard library

## Usage

The main implementation is in `fec.py`, which defines the `FluxEquilibriumClustering` class.

```python
from fec import FluxEquilibriumClustering

# X is a NumPy array of shape (n_samples, n_features)
model = FluxEquilibriumClustering()
model.fit(X)

# Cluster labels are determined by the implementation
# and stored according to the class interface.
```

The `fit` method accepts a two-dimensional NumPy array and computes a cluster assignment for each input point.

## Learning Objectives

This project explores graph-based clustering, nearest-neighbor methods, weighted graphs, iterative algorithms, numerical computation with NumPy, and cycle detection. It also demonstrates how local graph structure and iterative flow dynamics can be used to identify clusters in multidimensional data.

## Reference

DSC 395T: Algorithms and Data Structures, Programming Assignment #5 — Flux Equilibrium Clustering.
