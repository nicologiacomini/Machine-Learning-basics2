from get_data_info import GetDataInfo as gdi
import pandas as pd
import numpy as np

class GetMatrices:

    @staticmethod
    def get_matrices(threshold, theta):
        weight_matrix = gdi.get_weight_matrix(threshold, theta)
        x_matrix = gdi.get_data_from_csv()
        x_matrix = pd.DataFrame(x_matrix).to_numpy().astype(float)
        # normalize x_matrix
        x_matrix = (x_matrix - x_matrix.mean(axis=0)) / x_matrix.std(axis=0)
        n = weight_matrix.shape[0]

        diagonal_matrix = np.zeros((n, n))
        for i in range(n):
            val = 0
            for j in range(n):
                val += weight_matrix[i, j]
            diagonal_matrix[i, i] = val

        laplacian_matrix = diagonal_matrix - weight_matrix

        return weight_matrix, x_matrix, laplacian_matrix