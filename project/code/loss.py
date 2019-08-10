import numpy as np
from torch.autograd import Variable
from config import *
from typing import List


def calcLoss(featuresBatch: torch.Tensor, labelsBatch: np.ndarray):
    totalLoss = Variable(torch.Tensor([0]).type(double_type))
    batchSize = featuresBatch.shape[0]
    for sample in range(batchSize):
        clusterMeans, clusters = getClusters(featuresBatch[sample], labelsBatch[sample])
        totalLoss = totalLoss + lossParams.alpha * getVarLoss(clusterMeans, clusters)
        totalLoss = totalLoss + lossParams.beta * getDistLoss(clusterMeans)
        totalLoss = totalLoss + lossParams.gamma * getRegularizationLoss(clusterMeans)
    totalLoss = totalLoss / batchSize
    return totalLoss


def getVarLoss(clusterMeans: torch.Tensor, clusters: List[torch.Tensor]):
    """
    This function returns the inter-cluster variation loss
    The loss is minimized by clustering all the embeddings of the same instance together, up to some dv parameter
    :param clusterMeans: A Tensor of shape (N,C), where N is the number of instances (including the background) and C is
    the embedding dimension
    :param clusters: A list of Tensors. There are N Tensors in the list. each Tensor is of dimension (Pi,C) where Pi is
    the number of pixels in the i'th instance. C is the embedding dimension
    :return:
    The total variation loss of this image clusters
    """
    N = len(clusters) # number of clusters
    varLoss = Variable(torch.Tensor([0])).type(double_type)
    for c in range(N):
        mean = clusterMeans[c]
        cluster = clusters[c]
        M = cluster.shape[0]  # number of pixels
        varLoss = varLoss + torch.sum(
            torch.relu(
                torch.norm(cluster - mean, p=lossParams.norm, dim=1) - lossParams.dv
            )
            ** 2) \
                  / M
    varLoss = varLoss / N
    return varLoss


def getDistLoss(clusterMeans:torch.Tensor):
    """
    This function returns the distance loss between different clusters, up to some dd parameter.
    :param clusterMeans: the means of all clusters in the embedding space. shape is (N,C) where N is the number of
    instances (including background), and C is the embedding space dimension
    :return:
    the total distance loss
    """
    N = clusterMeans.shape[0]
    distLoss = Variable(torch.Tensor([0])).type(double_type)
    if N<2:
        return distLoss
    for cA in range(N):
        clusterA = clusterMeans[cA]
        for cB in range(cA+1,N):
            clusterB = clusterMeans[cB]
            distLoss = distLoss + torch.relu(2*lossParams.dd-torch.norm(clusterA-clusterB,p=lossParams.norm))**2

    distLoss = distLoss/(N*(N-1))
    return distLoss


def getRegularizationLoss(clusterMeans: torch.Tensor):
    """
    This function calculates the regularization loss. The regularization is on the cluster means
    :param clusterMeans: the means of all clusters in the embedding space. shape is (N,C) where N is the number of
    instances (including background), and C is the embedding space dimension
    :return: The total regularization loss calculated on the cluster means
    """
    regLoss = torch.mean(torch.norm(clusterMeans,dim=1,p=lossParams.norm))
    return regLoss


def getClusters(features: torch.Tensor, labels: np.ndarray):
    """
    This function performs clustering on the input features according to the true labels
    :param features: an (C,h,w) Tensor as outputted from the feature extractor
    :param labels: and (h,w) numpy.ndarray representing the true labels of each pixel. assuming 0 is the background and 255 is a boundry
    :return: a tuple T consists of the following:
    T[0] is a tensor of dimensions (K+1,C) - the clusters means (K clusters and background)
    T[1] is a list L with the following:
    L[0] = a tensor of dimensions (P0,C) representing the embeddings of all pixels the correspond to instance 0 (background)
    L[1] = a tensor of dimensions (P1,C) representing the embeddings of all pixels the correspond to instance 1
    ...
    L[K] = a tensor of dimensions (PK,C) representing the embeddings of all pixels the correspond to instance K
    """

    L = list()
    means = list()

    labels = labels.flatten()
    features = features.permute(1, 2, 0).contiguous()
    shape = features.size()
    features = features.view(shape[0] * shape[1], shape[2])

    instances, counts = np.unique(labels, return_counts=True)
    for instance, count in zip(instances, counts):
        if instance == 255:
            continue  # skip boundry
        locations = Variable(torch.LongTensor(np.where(labels == instance)[0]).type(long_type))
        vectors = torch.index_select(features, dim=0, index=locations).type(double_type)  # all vectors of this instance
        L.append(vectors)
        means.append(torch.mean(vectors, dim=0))
    T = (torch.stack(means), L)

    return T
