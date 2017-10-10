"""doc"""

import numpy as np

class SOFM:
    """Very basic implementation of an Self Organizing Feature Map"""
    def __init__(self, dim=3, m=100, n=100, learnrate=1.0, deltalr=0.9999, sigma=0.75):
        """doc"""
        self._weights = np.random.rand(m, n, dim)
        self._m = m
        self._n = n
        self._lr = learnrate
        self._delta_lr = deltalr
        self._sigma = sigma

        if self._weights.shape[2] == 2:
            for it_x in range(m):
                for it_y in range(n):
                    self._weights[it_x][it_y] = (it_x/m, it_y/n)

    def get_dim(self):
        """returns the user specified dimension of the weight vector per neuron"""
        return self._weights.shape[2]

    def get_weights(self):
        """get all the weights of all neurons"""
        return self._weights

    def set_vector(self, pos_x, pos_y, vector):
        """doc"""
        self.weights[pos_x][pos_y] = vector

    def dist(self, _a, _b):
        """calculate the distance of two vectors the same dimension"""
        return np.sqrt(np.sum((_a - _b) ** 2))

    def get_best_matching(self, inputvector):
        """doc"""
        delta_weights = self._weights - inputvector
        delta_weights = np.abs(delta_weights)
        delta_weights = np.sum(delta_weights, axis=2)
        min_pos = np.argmin(delta_weights)
        minx, miny = np.unravel_index(min_pos, delta_weights.shape)

        return minx, miny
    def gaussian(self, x, mu, sig):
        """doc"""
        return np.exp(-np.power((x - mu), 2.) / (2 * np.power(sig, 2.)))

    def train(self, inputvector):
        """doc"""
        bm_x, bm_y = self.get_best_matching(inputvector)
        #delta_vector = inputvector - self._weights[bm_x][bm_y]
        
        delta_matrix = inputvector - self._weights
        
        x1 = self.gaussian(np.linspace(0, 1, self._m), bm_x/(self._m - 1), self._sigma)
        x2 = self.gaussian(np.linspace(0, 1, self._n), bm_y/(self._n - 1), self._sigma)

        lis = []
        for i in range(self._weights.shape[2]):
            lis.append(np.outer(x1, x2))

        delta_weight = np.stack(lis, axis=2)
        delta_weight = delta_matrix * delta_weight * self._lr


        self._weights += delta_weight

        self._lr *= self._delta_lr
        return bm_x, bm_y
