import numpy as np
import math

class models_distance():

    def __init__(self,models_mean_set_1,models_covariance_set_1,models_mean_set_2,models_covariance_set_2):
        """

        :param models_mean_set_1:
        :param models_covariance_set_1:
        :param models_mean_set_2:
        :param models_covariance_set_2:
        """
        self.mean_set_1=models_mean_set_1
        self.covariance_set_1=models_covariance_set_1
        self.mean_set_2=models_mean_set_2
        self.covariance_set_2=models_covariance_set_2

    def covariance_distance(self):
        """

        :return:
        """
        covariance_dist = np.empty([self.covariance_set_1.shape[0], self.covariance_set_2.shape[0]])
        for c1 in range(0, self.covariance_set_1.shape[0]):
            for c2 in range(0, self.covariance_set_2.shape[0]):
                covariance_dist[c1, c2] = (np.linalg.norm(self.covariance_set_2[c2] - self.covariance_set_1[c1]))
        return covariance_dist

    def mean_distance(self):
        """

        :return:
        """
        mean_dist = np.empty([self.covariance_set_1.shape[0], self.covariance_set_2.shape[0]])
        for c1 in range(0, self.covariance_set_1.shape[0]):
            for c2 in range(0, self.covariance_set_2.shape[0]):
                mean_dist[c1, c2] = (np.linalg.norm(self.covariance_set_2[c2] - self.covariance_set_1[c1]))
        return mean_dist

    def KL_divergence(self):
        """
        Kullback–Leibler divergence
        :return:
        """
        kl_div = np.empty([self.covariance_set_1.shape[0], self.covariance_set_2.shape[0]])
        for c1 in range(0, self.covariance_set_1.shape[0]):
            for c2 in range(0, self.covariance_set_2.shape[0]):
                kl_div[c1, c2] =  0.5 * (
                    np.log(np.linalg.det(self.covariance_set_1[c1]) / np.linalg.det(self.covariance_set_2[c2]))
                    - self.mean_set_1.shape[1]
                    + np.trace(np.matmul(np.linalg.inv(self.covariance_set_1[c1]), self.covariance_set_2[c2]))
                    + np.matmul(np.matmul((self.mean_set_1[c1] - self.mean_set_2[c2]), np.linalg.inv(self.covariance_set_1[c1]))
                                , (self.mean_set_1[c1] - self.mean_set_2[c2])))
        return kl_div

    def Hellinger(self):
        """
        Hellinger distance
        :return:
        """
        Hel = np.empty([self.covariance_set_1.shape[0], self.covariance_set_2.shape[0]])
        for c1 in range(0, self.covariance_set_1.shape[0]):
            for c2 in range(0, self.covariance_set_2.shape[0]):
                Hel[c1, c2] = math.sqrt(
            1 - (np.linalg.det(self.covariance_set_2[c2]) * np.linalg.det(self.covariance_set_1[c1])) ** (1 / 4)
            / np.linalg.det(self.covariance_set_2[c2] / 2 + self.covariance_set_1[c1] / 2) ** (1 / 2)
            * math.exp(-np.matmul(np.matmul((self.mean_set_1[c1] - self.mean_set_2[c2]),
                                            np.linalg.inv(self.covariance_set_2[c2] / 2 + self.covariance_set_1[c1] / 2))
                                  , (self.mean_set_1[c1] - self.mean_set_2[c2])) / 8))
        return Hel
