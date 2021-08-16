# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 13:49:06 2021

@author: Jonathan Chien

Functions for cluster validation based on Density Based Clustering Validation
(DBCV) and Adjusted Mutual Information (AMI).
"""

# Standard Library imports.
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
import sklearn.metrics as sklm
import warnings

# Third party imports.
import hdbscan

# Local application imports.
from dbcvmaster.dbcv import compute_dbcv_score # dbcvmaster.dbcv = directory.package


def get_cos_dist(vectors):
    """
    Obtain pairwise cosine distance between observations (row vectors of an
    n_samples x n_features array).
    """
    cos_dist = sklm.pairwise_distances(vectors, metric='cosine')
    
    return cos_dist


def get_cos_dist2(vectors):
    """
    Calculate by hand pairwise cosine distance between observations 
    (row vectors of an n_samples x n_features array).
    """
    norms = np.sqrt((vectors**2).sum(axis=1))[:,np.newaxis]
    vectors = vectors/norms
    cos_dist = 1-(vectors @ vectors.T)
    
    return cos_dist


def grid_search_dbcv(vectors, min_clust_size_vals, min_samples_vals, 
                     dist_metric='cosine'):
    """
    Performs grid search for min_cluster_size and min_sample hyperparameters
    in HDBSCAN using the Density Based Cluster Validation (DBCV) index.
    Haversine has superior precision compared to versine (cos dist) for very
    small angles, though I'm not sure this is important here, as we are more 
    interested in big vs small distances, and not in comparing small distances
    with each other. Note as well that haversine metric is not supported by
    the scipy.spatial.distance.cdist function called in the dbcv function.

    Parameters
    ----------
    vectors : 2D array of float64
        m x n matrix of m observations in n-dim space (corresponding to n vars).
    min_clust_size_vals : range
        Values for min_cluster_size in HDBSCAN to be searched across.
    min_samples_vals : range
        Values for min_samples in HDBSCAN to be searched across.
    dist_metric : string
        Distance metric for HDBSCAN. Either 'cosine' or 'euclidean' (note that
        "cosine distance" is not a real distance metric as it fails to satisfy
        the triangle inequality).

    Returns
    -------
    results : dict
        'all_dbcv_scores' : 2D array of float64
            2D array of DBCV score corresponding to grid of hyperparameters 
            (one element for each combination of the two hyperparameters).
        'best_dbcv_score' : float64
            Maximum DBCV score across all combinations of hyperparameters.
        'best_min_clust_size' : int
            Optimal value for min_cluster_size in HDBSCAN.
        'best_min_samples' : int
            Optimal value for min_samples in HDBSCAN.    
    """
    # Needed in loop below if dist_metric == 'cosine'.
    cos_dist = get_cos_dist(vectors)
    
    # Iterate over all hyperparameter combinations and evaluate clustering 
    # using DBCV.
    dbcv_scores = np.full((len(min_clust_size_vals),len(min_samples_vals)), np.nan)
    for i_min_clust_size, min_clust_size in enumerate(min_clust_size_vals):
        for i_min_samples, min_samples in enumerate(min_samples_vals):
               
            # Display progress in console.
            print(f"Working on min_cluster_size = {min_clust_size} and "\
                  f"min_samples = {min_samples}")  
            
            # Cluster with HDBSCAN and evaluate with DBCV.
            if dist_metric == 'euclidean':
                clusterer = hdbscan.HDBSCAN(metric = dist_metric, 
                                            min_cluster_size=min_clust_size,
                                            min_samples=min_samples).fit(vectors)
                dbcv_scores[i_min_clust_size, i_min_samples] \
                    = compute_dbcv_score(vectors, clusterer.labels_)                                                                  
            elif dist_metric == 'cosine':
                clusterer = hdbscan.HDBSCAN(metric = 'precomputed', 
                                            min_cluster_size=min_clust_size,
                                            min_samples=min_samples).fit(cos_dist)
                dbcv_scores[i_min_clust_size, i_min_samples] \
                    = compute_dbcv_score(vectors, clusterer.labels_,
                                         dist_function=cosine)                                                            
            else:
                raise Exception("dist_metric must be 'euclidean or 'cosine'.")
    
    # Check for nans in output. 
    if (np.isnan(dbcv_scores)).sum() > 0:
        warnings.warn("At least one DBCV score has value nan.")
    
    # Store results for output.
    results = {}
    results['best_dbcv_score'] = np.max(dbcv_scores)
    max_dbcv_score_i = np.unravel_index(np.argmax(dbcv_scores, axis=None),
                                        dbcv_scores.shape)
    results['best_min_clust_size'] = min_clust_size_vals[max_dbcv_score_i[0]]  
    results['best_min_samples'] = min_samples_vals[max_dbcv_score_i[1]]
    results['all_dbcv_scores'] = dbcv_scores

    return results
    

def construct_dist_matrices(vectors, dist_metric, n_bootstraps=100,
                            subsample=0.85):
    """
    Returns all pairwise distances between b sets of m n-dim observations, 
    where each set of observations is generated by subsampling from original 
    data (without replacement).

    Parameters
    ----------
    vectors : 2D array-like
        m x n matrix of m observations in n-dim space (corresponding to n vars).
    dist_metric : string
        Distance metric used to calculate pairwise distances. Must be either
        'cosine' or 'euclidean'.
    n_bootstraps : int, optional
        Number of distance matrices to be generated. Each distance matrix is
        calculated based on a randomly subsampled version of the original data.
        The default is 100.
    subsample : int, optional
        Fraction of data to be sampled (without replacement) for the generation
        of each distance matrix. The default is 0.85.

    Raises
    ------
    Exception
        Exception raised if dist_metric is not specified as 'cosine' or 
        'euclidean', or if there are NaNs present in the data.

    Returns
    -------
    dist_matrices : 3D array of float64
        z x m x m matrix, where m is the number of n-dim observations in a 
        m x n matrix with n variables. dist_matrices is a "stack" of matrices,
        stacked along the first dim (z), with the second and third dims holding
        each matrix. Each element in each of these matrices is the distance
        between a point m_i and a point m_j in m = m_1 ... m_i. Diagonals of 
        each matrix should be zero, as it is the distance of a point to itself.
    """  
    # Check for NaNs in data.
    if np.sum((np.isnan(vectors))) > 0:
        raise Exception("NaNs present in data.")
            
    # Determine total number of observations, number of observations to remove
    # for subsampling, and number of dimensions (variables/features).
    n_obs = len(vectors[:,0])
    n_delete = int(np.rint((1-subsample)*n_obs))
    n_dims = len(vectors[0,:])
    
    # Create n_bootstraps subsampled versions of the dataset. sampled_vecs is
    # a stack (along the 1st dim) of matrices held in dims 2 and 3.
    sampled_vecs = np.full((n_bootstraps, n_obs-n_delete, n_dims), np.nan)
    for b in range(n_bootstraps):
        removed_neurons = np.random.choice(range(n_obs), n_delete, replace=False)
        sampled_vecs[b,:,:] = np.delete(vectors, removed_neurons, axis=0)
    
    # Calculate distance matrices for each subsample.
    if dist_metric == "cosine": 
        # Normalize rows of each matrix (dim 2 and 3) in sampled_vecs.
        norms = np.sqrt((sampled_vecs**2).sum(axis=2))[:,:,np.newaxis]
        sampled_vecs = sampled_vecs/norms
        
        # Transpose and do matrix multiplication ("elementwise" along 1st dim
        # of sampled_vecs) to get pairwise dot products among all rows in 
        # each of the respective matrices. Subtract from 1 to convert from cos
        # similarity to cos distance.
        dist_matrices = 1 - np.matmul(sampled_vecs,
                                      np.transpose(sampled_vecs, axes=(0,2,1)))    
    elif dist_metric == "euclidean":
        # Calculate each of the three sets (stacks) of component matrices.
        comp1 = (sampled_vecs**2).sum(axis=2)[:,:,np.newaxis]
        comp2 = (sampled_vecs**2).sum(axis=2)[:,np.newaxis,:]
        comp3 = -2 * np.matmul(sampled_vecs,
                               np.transpose(sampled_vecs, axes=(0,2,1)))
        
        # Diagonal of matrix sum of the three component matrices should be 0,
        # but some entries are off by small numerical error. Correct this error  
        # by setting those entries to zero. Take sqrt to obtain dist matrices.
        comp_sum = comp1 + comp2 + comp3
        comp_sum[np.abs(comp_sum) < 1E-12] = 0
        dist_matrices = np.sqrt(comp_sum)   
    else:
        raise Exception("dist_metric must be 'cosine' or 'euclidean'.")
        
    return dist_matrices


def grid_search_ami(vectors, min_clust_size_vals, min_samples_vals, 
                    dist_metric, n_bootstraps=100, subsample=0.85):
    """
    Performs grid search for optimal values for the min_cluster_size and 
    min_samples hyperparameters in HDBSCAN based on maximum Adjusted Mutual 
    Information score over subsamples of the data.

    Parameters
    ----------
    vectors : 2D array-like
        Data to be clustered. An m x n array of m observations in n-dimensional 
        space.
    min_clust_size_vals : range
        Values for min_cluster_size in HDBSCAN to be searched across.
    min_samples_vals : range
        Values for min_samples in HDBSCAN to be searched across.
    dist_metric : string
        Distance metric for HDBSCAN. 'cosine' or 'euclidean' (yeah, we know 
        cosine distance isn't a real distance metric).
    n_bootstraps : int, optional
        Number of bootstrap iterations for each hyperparameter combination. 
        The default is 100.
    subsample : int, optional
        Fraction of data to be sampled (without replacement) for each iteration
        during bootstrapping. The default is 0.85.

    Returns
    -------
    fx_results : dict
        Dictionary containing results of hyperparameter search.
        'max_ami' : float64
            Maximum adjusted mutual information score (mean across all
            bootstrap iterations) across all hyperparameter combinations.
        'best_min_clust_size' : int
            Optimal value for min_cluster_size in HDBSCAN.
        'best_min_samples' : int
            Optimal value for min_samples in HDBSCAN.
        'all_ami' : 2D array of float64
            Grid containing mean AMI (across bootstrapped subsamples) for each 
            combination of hyperparameters for HDBSCAN.
    """                        
    dist_matrices = construct_dist_matrices(vectors, dist_metric, 
                                           n_bootstraps, subsample)
    
    # Determine number of observations in subsample and of bootstrap iterations.
    n_obs = np.shape(dist_matrices)[1]
    n_bootstraps = np.shape(dist_matrices)[0]
    
    # Run HDBSCAN on precomputed distances for subsamples of data. Consider
    # using parfor loops but currently I see no way of vectorizing any part 
    # of this process or of otherwise expediting it, ugly as it is.
    all_ami_vals = np.zeros((len(min_clust_size_vals),len(min_samples_vals))) 
    for i_min_clust_size, min_clust_size in enumerate(min_clust_size_vals):        
        for i_min_samples, min_samples in enumerate(min_samples_vals):
            
            # Track progress in command window.
            print(f"Working on min_cluster_size = {min_clust_size} and "\
                  f"min_samples = {min_samples}")  
                
            cluster_labels = np.zeros((n_obs,n_bootstraps))  
            ami_vals = np.full((n_bootstraps,n_bootstraps), np.nan)
            for b in range(n_bootstraps):
                # Run HDBSCAN.
                clusterer = hdbscan.HDBSCAN(metric = 'precomputed', 
                                            min_cluster_size=min_clust_size,
                                            min_samples=min_samples)
                clusterer.fit(dist_matrices[b,:,:])                
                cluster_labels[:,b] = clusterer.labels_                
            
                # Calculate all pairwise adjusted mutual information scores as 
                # we go.
                for bb in range(b):
                    ami_vals[b,bb] \
                        = sklm.adjusted_mutual_info_score(cluster_labels[:,b],
                                                          cluster_labels[:,bb])
            
            # Calculate mean AMI across all subsamples for current 
            # hyperparameter combination.
            all_ami_vals[i_min_clust_size,i_min_samples] = np.nanmean(ami_vals)
    
    # Place results in dictionary for export.
    fx_results = {}
    fx_results['max_ami'] = np.max(all_ami_vals)
    max_ami_i = np.unravel_index(np.argmax(all_ami_vals, axis=None),
                                 all_ami_vals.shape)
    fx_results['best_min_clust_size'] = min_clust_size_vals[max_ami_i[0]]    
    fx_results['best_min_samples'] = min_samples_vals[max_ami_i[1]]
    fx_results['all_ami'] = all_ami_vals
    
    return fx_results


def plot_grid_search(index_scores, hp1_name, hp1_vals, hp2_name, hp2_vals):
    """
    Returns a heatmap of cluster validity score across the grid of 
    hyperparameters for HDBSCAN.

    Parameters
    ----------
    index_scores : 2D array-like
        2D array of clustering validator scores where each element corresponds
        to one cell on the grid of hyperparemeters.
    hp1_name : string
        Name of first hyperparameter. Used to label x axis of heatplot.
    hp1_vals : list
        List of values for the first hyperparmeter within the grid.
    hp2_name : string
        Name of second hyperparameter. Used to label y axis of heatplot.
    hp2_vals : list
        List of values for the second hyperparmeter within the grid..

    Returns
    -------
    heatmap : AxesImage
        Heatmap of cluster validity scores across grid of 
        hyperparameters.

    """    
    fig, ax = plt.subplots()
    plt.imshow(index_scores, interpolation='hamming', aspect = 'auto')
    # ax.set_xticks(hp1_vals)
    # ax.set_yticks(hp2_vals)
    plt.xlabel("min_samples")
    plt.ylabel("min_cluster_size")
    plt.colorbar()
    plt.show()
        