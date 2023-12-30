import numpy as np

class LDA_implemented:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = np.zeros(n_components)
        self._isfitted = False

    def __str__(self):
        return f"Linear discriminant analysis method.\nReturns {self.n_components} linear discriminants.\nLDA is {'' if self._isfitted else 'not '}fitted"

    def fit(self,X,y):
        N_CLASSES = len(np.unique(y))
        
        assert self.n_components <= N_CLASSES - 1, f"n_components can at most be k-1, the value provided was {self.n_components}"
        assert X.shape[0] == y.shape[0], f"Number of observations must be the same. X: {X.shape[0]}, y: {y.shape[0]}"

        mean_overall = np.mean(X, axis=0)
        S_B = 0
        S_W = 0
        for i in range(N_CLASSES):
            idx = np.where(y == i)[0]
            N = idx.shape[0]
            mean = np.mean(X[idx,:], axis=0)
            S_B += N * ((mean- mean_overall).reshape(-1,1) @ (mean - mean_overall).reshape(-1,1).T)
            S_W += (X[idx] - mean).T @ (X[idx] - mean)
        
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        sorted_indices = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues_sort = eigenvalues[sorted_indices]
        eigenvectors_sort = eigenvectors[:,sorted_indices].T
        self.linear_discriminants = eigenvectors_sort[:self.n_components]
        self._isfitted = True
        print("LDA succesfully fitted. To transform, call the transform method")

    def transform(self, X):
        assert self._isfitted, "LDA must be fitted before transforming"
        
        return np.real((X @ self.linear_discriminants.T))

    def fit_transform(self, X, y):
        N_CLASSES = len(np.unique(y))

        assert self.n_components <= N_CLASSES - 1, f"n_components can at most be k-1, the value provided was {self.n_components}"
        assert X.shape[0] == y.shape[0], f"Number of observations must be the same. X: {X.shape[0]}, y: {y.shape[0]}"
        mean_overall = np.mean(X, axis=0)
        S_B = 0
        S_W = 0
        for i in range(N_CLASSES):
            idx = np.where(y == i)[0]
            N = idx.shape[0]
            mean = np.mean(X[idx,:], axis=0)
            S_B += N * ((mean- mean_overall).reshape(-1,1) @ (mean - mean_overall).reshape(-1,1).T)
            S_W += (X[idx] - mean).T @ (X[idx] - mean)
        
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        sorted_indices = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues_sort = eigenvalues[sorted_indices]
        eigenvectors_sort = eigenvectors[:,sorted_indices].T
        self.linear_discriminants = eigenvectors_sort[:self.n_components]
        output = np.real((X @ self.linear_discriminants.T)) 
        self._isfitted = True
        print("LDA succesfully fitted and tranformed")
        return output
                