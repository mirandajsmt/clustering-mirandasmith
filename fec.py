from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels_``.
        """
        n = X.shape[0]
        
        # Step 1: Pre-compute
        # Compute global centroid
        centroid = X.mean(axis=0)
        
        # Calculate distances from each point to centroid
        distances = np.linalg.norm(X - centroid, axis=1)
        
        # Compute pairwise distances for sigma calculation if needed
        pairwise_dists = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        
        # Set sigma if not specified
        if self.sigma is None:
            # Get upper triangle indices to compute median of pairwise distances
            upper_tri_i, upper_tri_j = np.triu_indices(n, k=1)
            median_dist = np.median(pairwise_dists[upper_tri_i, upper_tri_j])
            sigma = 0.2 * median_dist
        else:
            sigma = self.sigma
        
        # Set effective k
        k_eff = min(self.k, n - 1)
        
        # Build k-NN graph with exponential weights
        # Store as (n, k) array with indices and weights
        knn_indices = np.zeros((n, k_eff), dtype=int)
        knn_weights = np.zeros((n, k_eff))
        
        for i in range(n):
            # Find k nearest neighbors (excluding self)
            dists_i = pairwise_dists[i].copy()
            dists_i[i] = np.inf  # Exclude self
            nearest_indices = np.argpartition(dists_i, k_eff)[:k_eff]
            
            # Sort by distance for consistency
            nearest_indices = nearest_indices[np.argsort(dists_i[nearest_indices])]
            
            # Compute weights
            weights = np.exp(-(pairwise_dists[i, nearest_indices]**2) / (sigma**2))
            
            knn_indices[i] = nearest_indices
            knn_weights[i] = weights
        
        # Store w_ij in sparse format (n, k)
        w_ij = knn_weights
        
        # Step 2: Initialize flux
        flux = np.ones(n)
        
        # Tolerance for relaxed downhill criterion
        tau = 0.1 * sigma
        
        # Step 3: Simulate flow
        delta = {}  # Store as sparse dict: (i,j) -> delta_ij
        
        for t in range(self.T):
            # Compute flux transport
            delta = {}
            
            for i in range(n):
                # Get neighbors from knn graph
                neighbors = knn_indices[i]
                neighbor_weights = knn_weights[i]
                
                # Find downhill neighbors (relaxed criterion)
                downhill_mask = distances[neighbors] <= distances[i] + tau
                downhill_neighbors = neighbors[downhill_mask]
                downhill_weights = neighbor_weights[downhill_mask]
                
                if len(downhill_neighbors) > 0:
                    # Compute flux-aware weights
                    if t == 0:
                        w_tilde = downhill_weights
                    else:
                        if self.beta is None:
                            phi_f = np.log(1 + flux[downhill_neighbors])
                        elif self.beta == 0:
                            phi_f = np.ones(len(downhill_neighbors))
                        else:
                            phi_f = flux[downhill_neighbors] ** self.beta
                        
                        w_tilde = downhill_weights * phi_f
                    
                    # Compute denominator
                    denom = np.sum(w_tilde)
                    
                    if denom > 0:
                        # Send flux to each downhill neighbor
                        for j, w in zip(downhill_neighbors, w_tilde):
                            delta[(i, j)] = self.alpha * w * flux[i] / denom
            
            # Update flux
            outgoing = np.zeros(n)
            incoming = np.zeros(n)
            
            for (i, j), d in delta.items():
                outgoing[i] += d
                incoming[j] += d
            
            flux = flux - outgoing + incoming
        
        # Step 4: Detect sinks
        outgoing_flux = np.zeros(n)
        for (i, j), d in delta.items():
            outgoing_flux[i] += d
        
        sink_indices = np.where(outgoing_flux < self.epsilon)[0]
        
        # If no sinks detected, use point closest to centroid
        if len(sink_indices) == 0:
            sink_indices = np.array([np.argmin(distances)])
        
        # Create boolean array for sinks
        sinks = np.zeros(n, dtype=int)
        sinks[sink_indices] = 1
        
        # Step 5: Assign clusters with cycle detection
        labels = -np.ones(n, dtype=int)
        
        # Assign labels to sinks
        sink_list = list(sink_indices)
        for idx, sink in enumerate(sink_list):
            labels[sink] = idx
        
        # For each non-sink, trace steepest path
        for i in range(n):
            if labels[i] >= 0:  # Already labeled (is a sink)
                continue
            
            path = [i]
            visited = {i}
            current = i
            
            while labels[current] < 0:
                # Get neighbors from knn graph
                neighbors = knn_indices[current]
                
                # Find downhill neighbors
                downhill_mask = distances[neighbors] <= distances[current] + tau
                candidates = neighbors[downhill_mask]
                
                if len(candidates) == 0:
                    # No downhill neighbor, make this a sink
                    new_sink_label = len(sink_list)
                    sink_list.append(current)
                    sinks[current] = 1
                    labels[current] = new_sink_label
                    break
                
                # Find steepest (max delta from last iteration)
                max_delta = -1
                steepest = None
                for j in candidates:
                    d = delta.get((current, j), 0)
                    if d > max_delta:
                        max_delta = d
                        steepest = j
                
                if steepest is None:
                    # No outgoing flux, make this a sink
                    new_sink_label = len(sink_list)
                    sink_list.append(current)
                    sinks[current] = 1
                    labels[current] = new_sink_label
                    break
                
                # Cycle detection
                if steepest in visited:
                    # Create new sink at current node
                    new_sink_label = len(sink_list)
                    sink_list.append(current)
                    sinks[current] = 1
                    labels[current] = new_sink_label
                    break
                
                path.append(steepest)
                visited.add(steepest)
                current = steepest
            
            # Assign all nodes in path to the terminal sink's cluster
            terminal_label = labels[current]
            for node in path:
                labels[node] = terminal_label
        
        # Step 6: Post-processing - merge small clusters
        s_min = max(2, n // 20)
        
        # Count cluster sizes
        unique_labels = np.unique(labels)
        cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
        
        # Identify small and large clusters
        small_clusters = [label for label, size in cluster_sizes.items() if size < s_min]
        large_clusters = [label for label, size in cluster_sizes.items() if size >= s_min]
        
        # Merge small clusters
        for small_label in small_clusters:
            if len(large_clusters) == 0:
                # No large clusters to merge into, keep as is
                break
            
            # Compute centroid of small cluster
            small_cluster_points = X[labels == small_label]
            mu_small = small_cluster_points.mean(axis=0)
            
            # Find nearest large cluster
            min_dist = float('inf')
            nearest_large = None
            
            for large_label in large_clusters:
                large_cluster_points = X[labels == large_label]
                mu_large = large_cluster_points.mean(axis=0)
                dist = np.linalg.norm(mu_small - mu_large)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_large = large_label
            
            # Merge small into large
            labels[labels == small_label] = nearest_large

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"