class LossParams:
    def __init__(self, dv=2, dd=10, alpha=1, beta=1, gamma=1, delta=0.001,
                 include_edges=True, edges_max_pixels=200,
                 norm=2, include_weighted_mean=False,
                 center_weight=0.1, corners_weight=0.025):
        """
        :param dv: variance-term (pulls the embedding towards the mean)
        :param dd: distance-term (pushes clusters away from each other)
        :param alpha: regularization for the variance-term
        :param beta: regularization for the distance-term
        :param delta: regularization term for the cluster's mean
        :param include_edges: boolean flag whether or not to include the object's edges in the loss
        :param edges_max_pixels: max number of pixels to take from the edges
               (only applies when include_edges is True)
        :param norm: norm type to use when computing vectors distances in the embedding space
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.include_edges = include_edges
        self.edges_max_pixels = edges_max_pixels
        self.gamma = gamma
        self.norm = norm
        self.dv = dv
        self.dd = dd
        self.include_weighted_mean = include_weighted_mean
        self.center_weight = center_weight
        self.corners_weight = corners_weight
