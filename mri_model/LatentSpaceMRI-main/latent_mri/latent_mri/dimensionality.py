import numpy as np

from skdim.id import lPCA, MLE
from sklearn.decomposition import IncrementalPCA


class Dimensionality:
    @staticmethod
    def effective_dimensionality(data, ver='participation_ratio'):
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        lpca = lPCA(ver=ver).fit(data)
        return lpca.dimension_

    @staticmethod
    def effective_dimensionality_batch(data, batch_size=48):
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)

        num_data = data.shape[0]

        assert num_data % batch_size == 0
        num_batches = num_data // batch_size

        ipca = IncrementalPCA(n_components=num_batches)
        for i_batch in range(num_batches):
            start = i_batch * batch_size
            end = start + batch_size
            ipca.partial_fit(data[start:end])

        singular_values = ipca.singular_values_
        eigenvalues = singular_values ** 2
        eigenvalues = eigenvalues / np.sum(eigenvalues)

        return np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)

    @staticmethod
    def intrinsic_dimensionality(data, num_neighbors=20, num_jobs=-1):
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        mle = MLE().fit(data, n_neighbors=num_neighbors, n_jobs=num_jobs)
        return mle.dimension_
