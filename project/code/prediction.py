import hdbscan
import numpy as np
import skimage.measure
from skimage.transform import rescale
from sklearn.decomposition import PCA

from mrf_wrapper import denoise_colored_image


def predict_label(features, downsample_factor=1, min_cluster=10):
    '''
    predicts a segmentation mask from the network output. Uses PCA to reduce dimesionality
    of the input, mainly due to performance reasons.
    :param features: (c,h,w) ndarray containing the feature vectors outputted by the model
    :param downsample_factor: the features are downsampled by this factor using max-pooling.
                        this improves stability of the clustering, and reduces running time.
    :param min_cluster: hDBSCAN minimal cluster size. It is recommended to tune this hyperparametr
                    as it has great effect on the clustering results.
    :return: (h,w) ndarray with the predicted label (currently without class predictions
    '''
    features = np.transpose(features, [1,2,0])  # transpose to (h,w,c)
    features = skimage.measure.block_reduce(features, (downsample_factor,downsample_factor,1), np.max) #reduce resolution for performance

    h = features.shape[0]
    w = features.shape[1]
    c = features.shape[2]

    flat_features = np.reshape(features, [h*w,c])
    # reduced_features = reduce(flat_features, 10)  # reduce dimension using PCA
    reduced_features = flat_features  # reduce dimension using PCA
    cluster_mask = cluster_features(reduced_features, min_cluster) # cluster with hDBSCAN
    #cluster_mask = determine_background(flat_features, cluster_mask)
    predicted_label = np.reshape(cluster_mask, [h,w])

    predicted_label = denoise_colored_image(predicted_label)

    predicted_label = rescale(predicted_label, order=0, mode='constant', scale=downsample_factor,
                              preserve_range=True)
    return np.asarray(predicted_label, np.int32)


def cluster_features(features, min_cluster):
    '''
    this function takes a (h*w,c) numpy array, and clusters the c-dim points using MeanShift/DBSCAN.
    this function is meant to use for visualization and evaluation_metrics only.  Meanshift is much slower
    but yields significantly better results.
    :param features: (h*w,c) array of h*w d-dim features extracted from the photo.
    :param min_cluster: min_cluster: hDBSCAN minimal cluster size. It is recommended to tune this hyperparametr
                    as it has great effect on the clustering results.
    :return: returns a (h*w,1) array with the cluster indices.
    '''
    # Define DBSCAN instance and cluster features
    dbscan = hdbscan.HDBSCAN(algorithm='boruvka_kdtree',min_cluster_size=min_cluster)
    labels = dbscan.fit_predict(features) + 1  # Unclustered pixels are labeled as -1 by the hDBSCAN

    return labels


def reduce(features, dimension=10):
    '''
    performs PCA dimensionality reduction on the input features
    :param features: a (n, d) or (h,w,d) numpy array containing the data to reduce
    :param dimension: the number of output channels
    :return: a (n, dimension) numpy array containing the reduced data.
    '''
    pca = PCA(n_components=dimension)
    pca_results = pca.fit_transform(features)
    return pca_results
