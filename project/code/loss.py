import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import config
import torch
from utils.image_helper import get_image_center_pixel, get_image_mask_corners_pixels


class LossModule(nn.Module):
    def __init__(self, loss_params):
        self.loss_params = loss_params
        super(LossModule, self).__init__()

    def forward(self, features, labels, labelEdges):
        return self.calcLoss(features, labels, labelEdges)

    def calcLoss(self, featuresBatch, labelsBatch, labelEdgesBatch):
        totalLoss = Variable(torch.Tensor([0]).type(config.double_type))
        varLoss = Variable(torch.Tensor([0]).type(config.double_type))
        distLoss = Variable(torch.Tensor([0]).type(config.double_type))
        edgeLoss = Variable(torch.Tensor([0]).type(config.double_type))
        regLoss = Variable(torch.Tensor([0]).type(config.double_type))
        batchSize = featuresBatch.shape[0]
        for sample in range(batchSize):
            clusterMeans, clusters = self.getClusters(featuresBatch[sample], labelsBatch[sample], labelEdgesBatch[sample])
            varLoss = varLoss + self.loss_params.alpha * self.getVarLoss(clusterMeans, clusters)
            distLoss = distLoss + self.loss_params.beta * self.getDistLoss(clusterMeans, clusters)
            edgeLoss = edgeLoss + self.loss_params.gamma * self.getEdgesLoss(clusterMeans, clusters)
            regLoss = regLoss + self.loss_params.delta * self.getRegularizationLoss(clusterMeans)

        totalLoss = varLoss + distLoss + edgeLoss + regLoss
        return totalLoss, varLoss, distLoss, edgeLoss, regLoss

    def getVarLoss(self, clusterMeans, clusters):
        """
        This function returns the inter-cluster variation loss
        The loss is minimized by clustering all the embeddings of the same instance together, up to some dv parameter
        :param clusterMeans: A Tensor of shape (N,C), where N is the number of instances (including the background) and C is
        the embedding dimension
        :param clusters: A list of 2-tuples. index 0 of the 2-tuple is the embeddings of the instance pixels, the
        dimensions are (PiA,C) (there are PiA pixels in object i). index 1 of the 2-tuple is the embeddings of the instance
        boundary pixels, the dimensions are (PiB,C) (there are PiB pixels in the boundary of object i). C is the embedding
        dimension There are N 2-tuples in the list.
        :return:
        The total variation loss of this image clusters
        """
        N = len(clusters)  # number of clusters
        varLoss = Variable(torch.Tensor([0])).type(config.double_type)
        for c in range(N):
            mean = clusterMeans[c]
            cluster = clusters[c][0]
            M = cluster.shape[0]  # number of pixels
            varLoss = varLoss + torch.sum(
                torch.relu(
                    torch.norm(cluster - mean, p=self.loss_params.norm, dim=1) - self.loss_params.dv
                )
                ** 2) \
                      / M
        varLoss = varLoss / N
        return varLoss

    def getDistLoss(self, clusterMeans, clusters):
        """
        This function returns the distance loss between different clusters, up to some dd parameter.
        :param clusterMeans: A Tensor of shape (N,C), where N is the number of instances (including the background) and C is
        the embedding dimension
        :param clusters: A list of 2-tuples. index 0 of the 2-tuple is the embeddings of the instance pixels, the
        dimensions are (PiA,C) (there are PiA pixels in object i). index 1 of the 2-tuple is the embeddings of the instance
        boundary pixels, the dimensions are (PiB,C) (there are PiB pixels in the boundary of object i). C is the embedding
        dimension There are N 2-tuples in the list.
        :return:
        the total distance loss
        """
        N = clusterMeans.shape[0]
        distLoss = Variable(torch.Tensor([0])).type(config.double_type)
        if N < 2:
            return distLoss
        for cA in range(N):
            clusterAMean = clusterMeans[cA]
            for cB in range(cA + 1, N):
                clusterBMean = clusterMeans[cB]
                # making sure that the clusters centers are far from each other by at least 2*dd
                distLoss = distLoss + torch.relu(
                    2 * self.loss_params.dd - torch.norm(clusterAMean - clusterBMean, p=self.loss_params.norm)
                ) ** 2

        distLoss = distLoss / (N * (N - 1))
        return distLoss

    def getEdgesLoss(self, clusterMeans, clusters):
        N = clusterMeans.shape[0]
        edgeLoss = Variable(torch.Tensor([0])).type(config.double_type)
        if N < 2 or not self.loss_params.include_edges:
            return edgeLoss

        for cA in range(N):
            clusterAEdges = clusters[cA][1]
            for cB in range(cA + 1, N):
                clusterBEdges = clusters[cB][1]
                clusterAEdgesRepeated = clusterAEdges.repeat(clusterBEdges.shape[0], 1)
                clusterBEdgesRepeated = clusterBEdges.repeat(1, clusterAEdges.shape[0]).view(
                    clusterAEdgesRepeated.shape[0],
                    -1)
                # making sure that that the edges of each instance are at least 2(dd-dv) apart
                edgeLoss = edgeLoss + torch.sum(torch.relu(
                    2 * (self.loss_params.dd - self.loss_params.dv) - torch.norm(
                        clusterAEdgesRepeated - clusterBEdgesRepeated,
                        p=self.loss_params.norm, dim=1)) ** 2) / \
                           clusterAEdgesRepeated.shape[0]

        edgeLoss = edgeLoss / (N * (N - 1))
        return edgeLoss

    def getRegularizationLoss(self, clusterMeans):
        """
        This function calculates the regularization loss. The regularization is on the cluster means
        :param clusterMeans: the means of all clusters in the embedding space. shape is (N,C) where N is the number of
        instances (including background), and C is the embedding space dimension
        :return: The total regularization loss calculated on the cluster means
        """
        regLoss = torch.mean(torch.norm(clusterMeans, dim=1, p=self.loss_params.norm))
        return regLoss

    def getClusters(self, features, labels, labelEdges):
        """
        This function performs clustering on the input features according to the true labels
        :param features: an (C,h,w) Tensor as outputted from the feature extractor
        :param labels: an (h,w) numpy.ndarray representing the true labels of each pixel. assuming 0 is the background and 255 is a boundry
        :param labelEdges: an (h,w) numpy.ndarray representing the true labels of each pixel but only on the boundaries, assuming 255 is non boundary.
        :return: a tuple T consists of the following:
        T[0] is a tensor of dimensions (K+1,C) - the clusters means (K clusters and background)
        T[1] is a list L with the following:
        L[0] = a 2-tuple: index 0 - a tensor of dimensions (P0A,C) representing the embeddings of all pixels the correspond to instance 0 (background)
                          index 1 - a tensor of dimensions (P0B,C) representing the embeddings of the boundary pixels of instance 0 (background)
        L[1] = a 2-tuple: index 0 - a tensor of dimensions (P1A,C) representing the embeddings of all pixels the correspond to instance 1
                          index 1 - a tensor of dimensions (P1B,C) representing the embeddings of the boundary pixels of instance 1
        ...
        L[K] = a 2-tuple: index 0 - a tensor of dimensions (PKA,C) representing the embeddings of all pixels the correspond to instance 1
                          index 1 - a tensor of dimensions (PKB,C) representing the embeddings of the boundary pixels of instance 1

        """

        L = list()
        means = list()

        if self.loss_params.include_weighted_mean:
            label_corners, label_centers = self.get_weighted_pixels_from_label(labels)
            label_corners = label_corners.flatten()
            label_centers = label_centers.flatten()

        labels = labels.flatten()
        labelEdges = labelEdges.flatten()
        features = features.permute(1, 2, 0).contiguous()
        shape = features.size()
        features = features.view(shape[0] * shape[1], shape[2])

        instances, counts = np.unique(labels, return_counts=True)
        for instance, count in zip(instances, counts):
            if instance == config.PIXEL_IGNORE_VAL:
                continue  # skip boundry for VOC

            vectors = self._get_instance_vectors(features, labels, instance)
            boundary_vectors = self._get_instance_vectors(features, labelEdges, instance,
                                                          limit=self.loss_params.edges_max_pixels)
            L.append((vectors, boundary_vectors))

            if self.loss_params.include_weighted_mean:
                mean = self._calc_weighted_mean(features, instance, label_centers, label_corners, vectors)
            else:
                mean = torch.mean(vectors, dim=0)
            means.append(mean)

        T = (torch.stack(means), L)

        return T

    def _calc_weighted_mean(self, features, instance, label_centers, label_corners, vectors):
        corners_vectors = self._get_instance_vectors(features, label_corners, instance)
        centers_vectors = self._get_instance_vectors(features, label_centers, instance)

        number_of_pixels = len(vectors)

        center_weight_for_instance = int(np.ceil(self.loss_params.center_weight * number_of_pixels))
        corners_weight_for_instance = int(np.ceil(self.loss_params.corners_weight * number_of_pixels))

        centers_vectors_torch = torch.cat([centers_vectors] * center_weight_for_instance)
        corners_vectors_torch = torch.cat([corners_vectors] * corners_weight_for_instance)

        mean = torch.mean(torch.cat([vectors, centers_vectors_torch, corners_vectors_torch]), dim=0)

        return mean

    @staticmethod
    def _get_instance_vectors(features, ground_truth, instance, limit=None):
        locations = Variable(torch.LongTensor(np.where(ground_truth == instance)[0]).type(config.long_type))

        if limit is not None:
            if locations.shape[0] > limit:
                locations = Variable(torch.LongTensor(limit).random_(0, locations.shape[0]).type(config.long_type))

        # all vectors of this instance
        vectors = torch.index_select(features, dim=0, index=locations).type(config.double_type)

        return vectors

    @staticmethod
    def get_weighted_pixels_from_label(label):
        label_corners = np.full(label.shape, config.PIXEL_IGNORE_VAL)
        label_centers = np.full(label.shape, config.PIXEL_IGNORE_VAL)

        instances = np.unique(label)
        for instance in instances:
            coords = get_image_mask_corners_pixels(label, instance)
            for x, y in coords:
                label_corners[x][y] = instance

            center = get_image_center_pixel(label, instance)
            label_centers[center[0]][center[1]] = instance

        return label_corners, label_centers
