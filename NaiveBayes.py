from scipy.stats import gaussian_kde
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


 
class NB_clf:
    def __init__(self, bandwidth=0.5, pdf_method='kde'):
        self.bandwidth = bandwidth
        self.pdf_method = pdf_method

        # Instance attributes
        self.KDE_dict = {} 
        self.class_priors = []
        self.n_classes = 0
        self.n_features = 0
        self.n_total = 0
        self.is_fitted = False

    def fit(self, X, y):
        '''First the class priors are being computed.
        Then a Kernel Density Estimate (KDE) is created
        for each feature within each class.'''
        self.n_total = len(X)
        self.n_classes = len(np.unique(X))
        self.n_features = y.shape[1]

        for k in range(self.n_classes):
            idx = np.where(y==k)[0]
            X_class = X[idx]
            n = len(X_class)
            self.class_priors.append(n / self.n_total)

            class_KDE = []
            for feature in range(self.n_features):
                X_class_feat = X_class[:, feature]
                KDE = gaussian_kde(X_class_feat, bw_method=self.bandwidth)
                class_KDE.append(KDE)

            self.KDE_dict[k] = class_KDE

        self.is_fitted = True


    def predict(self, X):
        '''Make predictions on an array 
        of new observations'''

        assert self.is_fitted, 'Naive Bayes Classifier must be fitted first'

        predictions = []
        n_total = len(X)
        for i in range(n_total):
            obs = X[i, :]
            class_likelihoods = []

            for k in range(self.n_classes):
                marginals = []

                for feature in range(self.n_features):
                    obs_val = obs[feature]
                    KDE = self.KDE_dict[k][feature]
                    marginal = KDE.pdf(obs_val)
                    marginals.append(marginal)

                marginals_prod = np.prod(marginals)
                class_likelihoods.append(self.class_priors[k] * marginals_prod)
            pred_idx = class_likelihoods.index(max(class_likelihoods))
            predictions.append(pred_idx)
        
        return np.array(predictions)
    

    def plot_KDEs(self, X, y, class_dict, feature_labels, colors):
        '''Plots the kernel density estimates in a 
        1xk grid, where k is the number of classes,
        with the KDE for each feature in the same plot'''

        assert self.is_fitted, 'Naive Bayes Classifier must be fitted first'

        fig, axs = plt.subplots(nrows=1, ncols=self.n_classes, figsize = [16,3.5])
        
        for k in range(self.n_classes):
            idx = np.where(y==k)[0]
            class_data = X[idx][:,:self.n_features]
            min_val = np.min(class_data)
            max_val = np.max(class_data)
            points = np.linspace(max_val, min_val)

            for feature in range(self.n_features):
                KDE = self.KDE_dict[k][feature]
                KDE_y = KDE.pdf(points)
                feature_name = feature_labels[feature]
                axs[k].plot(points, KDE_y, label=feature_name, color=colors[feature])
        
            class_name = class_dict[k]
            axs[k].set_title(f'KDEs for {class_name} class')
            axs[k].legend(loc='upper right', fontsize=9)

        fig.tight_layout()
    
        