"""
Synthetic data generation for KD-tree prefilter validation.

Generates:
- Gaussian catalogs: clustered LEO-like RSOs with anisotropic covariances
- GMM catalogs: multi-component objects with controlled overlap
"""

from typing import List, Tuple

import numpy as np

from .gmm_hellinger import GMM


def generate_gaussian_catalog(n_objects: int = 100,
                              n_clusters: int = 5,
                              cluster_spread: float = 1000.0,  # km
                              base_cov_scale: float = 1.0,  # km^2
                              anisotropy: float = 0.1,
                              seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic LEO-like Gaussian RSO catalog.
    
    Creates n_clusters of objects, each cluster centered at a random LEO position.
    Covariances are anisotropic with controlled orientation.
    
    Parameters
    ----------
    n_objects : int
        Total number of objects to generate
    n_clusters : int
        Number of spatial clusters
    cluster_spread : float
        Standard deviation of cluster centers (km)
    base_cov_scale : float
        Base covariance scale (km^2)
    anisotropy : float
        Ratio of smallest to largest eigenvalue of covariance (0 < anisotropy <= 1)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    list of (mean, covariance) tuples
        Each mean is shape (3,), each covariance is shape (3, 3)
    """
    rng = np.random.default_rng(seed)
    
    # LEO-like orbit: roughly circular at ~500-800 km altitude
    # Use Earth-centered coordinates
    earth_radius = 6371.0  # km
    orbit_radius = earth_radius + rng.uniform(400, 800, n_clusters)
    
    objects = []
    objects_per_cluster = n_objects // n_clusters
    remainder = n_objects % n_clusters
    
    for c in range(n_clusters):
        # Random orbital plane
        inclination = rng.uniform(0, np.pi)
        raan = rng.uniform(0, 2*np.pi)
        arg_peri = rng.uniform(0, 2*np.pi)
        
        # Cluster center on orbital plane
        true_anomaly = rng.uniform(0, 2*np.pi)
        
        # Position in orbital plane
        x_orb = orbit_radius[c] * np.cos(true_anomaly)
        y_orb = orbit_radius[c] * np.sin(true_anomaly)
        z_orb = 0.0
        
        # Rotate to ECI frame
        cos_i, sin_i = np.cos(inclination), np.sin(inclination)
        cos_raan, sin_raan = np.cos(raan), np.sin(raan)
        cos_ap, sin_ap = np.cos(arg_peri), np.sin(arg_peri)
        
        # Rotation matrix
        R = np.array([
            [cos_raan*cos_ap - sin_raan*cos_i*sin_ap,
             -cos_raan*sin_ap - sin_raan*cos_i*cos_ap,
             sin_raan*sin_i],
            [sin_raan*cos_ap + cos_raan*cos_i*sin_ap,
             -sin_raan*sin_ap + cos_raan*cos_i*cos_ap,
             -cos_raan*sin_i],
            [sin_i*sin_ap,
             sin_i*cos_ap,
             cos_i]
        ])
        
        pos_orb = np.array([x_orb, y_orb, z_orb])
        cluster_center = R @ pos_orb
        
        n_this_cluster = objects_per_cluster + (1 if c < remainder else 0)
        
        for _ in range(n_this_cluster):
            # Position: small perturbation from cluster center
            pos = cluster_center + rng.normal(0, cluster_spread/10, 3)
            
            # Anisotropic covariance
            # Principal axes aligned with radial, along-track, cross-track
            # Radial has largest uncertainty, cross-track smallest
            evals = base_cov_scale * np.array([
                1.0,                    # radial
                0.5,                    # along-track
                anisotropy              # cross-track
            ])
            
            # Random orientation
            Q, _ = np.linalg.qr(rng.normal(0, 1, (3, 3)))
            # Ensure proper rotation (det = 1)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            
            Sigma = Q @ np.diag(evals) @ Q.T
            
            # Ensure symmetric
            Sigma = 0.5 * (Sigma + Sigma.T)
            
            objects.append((pos, Sigma))
    
    return objects


def generate_gmm_catalog(n_objects: int = 50,
                         n_components_range: Tuple[int, int] = (2, 4),
                         component_overlap: float = 0.5,
                         seed: int = 42) -> List[GMM]:
    """
    Generate synthetic GMM catalog with controlled overlap.
    
    Creates objects with multiple Gaussian components.
    Overlap parameter controls how much components from different objects
    overlap - higher means more ambiguity in probabilistic ranking.
    
    Parameters
    ----------
    n_objects : int
        Number of objects to generate
    n_components_range : tuple
        (min, max) components per object
    component_overlap : float
        Controls component spread (0 = well separated, 1 = highly overlapping)
    seed : int
        Random seed
    
    Returns
    -------
    list of GMM objects
    """
    rng = np.random.default_rng(seed)
    
    objects = []
    
    for _ in range(n_objects):
        n_comp = rng.integers(n_components_range[0], n_components_range[1] + 1)
        
        # Base position
        base_pos = rng.uniform(-5000, 5000, 3)
        
        weights = []
        means = []
        covs = []
        
        for c in range(n_comp):
            # Weight: Dirichlet distribution
            weight = rng.random()
            weights.append(weight)
            
            # Component position: small perturbation from base
            # overlap controls spread
            pos = base_pos + rng.normal(0, 100 * (1 + component_overlap), 3)
            means.append(pos)
            
            # Covariance
            evals = rng.uniform(10, 1000 * (1 + component_overlap), 3)
            Q, _ = np.linalg.qr(rng.normal(0, 1, (3, 3)))
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            Sigma = Q @ np.diag(evals) @ Q.T
            Sigma = 0.5 * (Sigma + Sigma.T)
            covs.append(Sigma)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        means = np.array(means)
        covs = np.array(covs)
        
        gmm = GMM(weights, means, covs)
        objects.append(gmm)
    
    return objects


def generate_mixed_catalog(n_gaussian: int = 80,
                           n_gmm: int = 20,
                           **kwargs) -> List:
    """
    Generate mixed catalog with both Gaussian and GMM objects.
    """
    gaussians = generate_gaussian_catalog(n_gaussian, seed=kwargs.get('seed', 42))
    gmms = generate_gmm_catalog(n_gmm, seed=kwargs.get('seed', 43))
    return gaussians + gmms


def generate_test_case_identical(n: int = 10,
                                  seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate test case where Euclidean and Hellinger rankings coincide.
    
    Uses identical isotropic covariances for all objects.
    """
    rng = np.random.default_rng(seed)
    
    objects = []
    for _ in range(n):
        mu = rng.uniform(-1000, 1000, 3)
        Sigma = np.eye(3) * 100.0  # Identical isotropic
        objects.append((mu, Sigma))
    
    return objects


def generate_test_case_anisotropic(n: int = 10,
                                    seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate test case with anisotropic covariances where
    Hellinger ranking differs from Euclidean.
    """
    rng = np.random.default_rng(seed)
    
    objects = []
    for _ in range(n):
        mu = rng.uniform(-1000, 1000, 3)
        
        # Strongly anisotropic covariance
        evals = np.array([1000.0, 100.0, 10.0])  # 100:1 ratio
        Q, _ = np.linalg.qr(rng.normal(0, 1, (3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        Sigma = Q @ np.diag(evals) @ Q.T
        Sigma = 0.5 * (Sigma + Sigma.T)
        
        objects.append((mu, Sigma))
    
    return objects
