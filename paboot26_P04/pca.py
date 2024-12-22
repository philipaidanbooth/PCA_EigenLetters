'''pca_cov.py
Performs principal component analysis using the covariance matrix of the dataset
Philip Booth
CS 251 / 252: Data Analysis and Visualization
Fall 2024
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_transformations import normalize, center


class PCA:
    '''Perform and store principal component analysis results

    NOTE: In your implementations, only the following "high level" `scipy`/`numpy` functions can be used:
    - `np.linalg.eig`
    The numpy functions that you have been using so far are fine to use.
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        # orig_means: ndarray. shape=(num_selected_vars,)
        #   Means of each orignal data variable
        self.orig_means = None

        # orig_mins: ndarray. shape=(num_selected_vars,)
        #   Mins of each orignal data variable
        self.orig_mins = None

        # orig_maxs: ndarray. shape=(num_selected_vars,)
        #   Maxs of each orignal data variable
        self.orig_maxs = None
        
        self.optimal_index = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        
        data = center(data)
        N,M = data.shape
        cov_matrix = (data.T @ data)/(N-1)

        return cov_matrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        return list(e_vals/e_vals.sum())

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        return list(np.cumsum(prop_var))

    def fit(self, vars, normalize_dataset=False):
        '''Fits PCA to the data variables `vars` by computing the full set of PCs. The goal is to compute 
        - eigenvectors and eigenvalues
        - proportion variance accounted for by each PC.
        - cumulative variance accounted for by first k PCs.
        
        Does NOT actually transform data by PCA.

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize_dataset: boolean.
            If True, min-max normalize each data variable it ranges from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.
        
        HINT:
        - It may be easier to convert to numpy ndarray format once selecting the appropriate data variables.
        - Before normalizing (if normalize_dataset is true), create instance variables containing information that would
        be needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        - Remember, this method does NOT actually transform the dataset by PCA.
        '''
        usable_vars = []

        for var in vars:
            if var in self.data.columns:
                usable_vars.append(var)
        self.vars = usable_vars

        selected_data = self.data[usable_vars].to_numpy()

        if normalize_dataset:
            self.orig_means = np.mean(selected_data, axis=0)
            self.orig_maxs = np.max(selected_data, axis=0)
            self.orig_mins = np.min(selected_data, axis=0)

            selected_data = (selected_data - self.orig_mins) / (self.orig_maxs - self.orig_mins)
            self.normalized = True

        self.A = selected_data
        
        
        covariance_matrix = self.covariance_matrix(selected_data)
        e_vals, e_vecs = np.linalg.eig(covariance_matrix)
        idx = np.argsort(e_vals)[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]

        self.e_vals = e_vals 
        self.e_vecs = e_vecs

        prop_var = self.compute_prop_var(e_vals)
        cum_var = self.compute_cum_var(prop_var)
        self.prop_var = prop_var
        self.cum_var = cum_var
        

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if num_pcs_to_keep is None:
            num_pcs_to_keep = len(self.cum_var)
            
            
        # if self.cum_var is None or num_pcs_to_keep > len(self.cum_var):
        #     num_pcs_to_keep = len(self.cum_var)
        
        x = np.arange(1, num_pcs_to_keep+1)
        y = self.cum_var[:num_pcs_to_keep]
        plt.plot(x,y, marker = 'o')
 

        plt.xlabel('Number of Principle Components')
        plt.ylabel('Explained Variance in the model')
        plt.title("Elbow Plot")
        

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        
        eigs = self.get_eigenvectors()[:,pcs_to_keep]
        self.A_proj = center(self.A) @ eigs
        return self.A_proj
        #goal is to take a list of pc's to keep and spit out an ndarray

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        (Week 2)

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars). Data projected onto top K PCs then projected back to data space.

        NOTE: If you normalized, remember to rescale the data projected back to the original data space.
        '''
        
        center_data = center(self.A)
        top_k_eigenvectors = self.e_vecs[:, :top_k]
        #pca_proj = self.pca_project(list(range(top_k)))
        proj_data = np.dot(center_data, top_k_eigenvectors)
        data_proj_back = np.dot(proj_data, top_k_eigenvectors.T)

        mean_vect = np.mean(self.A, axis=0)
        
        reconstructed = data_proj_back + mean_vect

        return reconstructed

    def loading_plot(self):
        '''Create a loading plot of the top 2 PC eigenvectors

        (Week 2)

        TODO:
        - Plot a line joining the origin (0, 0) and corresponding components of the top 2 PC eigenvectors.
            Example: If e_0 = [0.1, 0.3] and e_1 = [1.0, 2.0], you would create two lines to join
            (0, 0) and (0.1, 1.0); (0, 0) and (0.3, 2.0).
            Number of lines = num_vars
        - Use plt.annotate to label each line by the variable that it corresponds to.
        - Reminder to create useful x and y axis labels.
        '''
        fig, axs = plt.subplots(figsize=(7, 7))

        
        eigs2 = self.get_eigenvectors()[:, :2]
        num_vars = eigs2.shape[0]
        print(num_vars)
        headers = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
        origin = np.array([0,0])
        for i in range(num_vars):
            x,y = eigs2[i]
            axs.plot([origin[0], x], [origin[1], y], lw=1.5)  # Line from origin to (x, y)
            #axs.annotate(f'Var{i+1}', xy=(x, y), xytext=(x, y), fontsize=10)
            axs.annotate(headers[i], xy=(x,y) )
        axs.set_xlabel('PC1')
        axs.set_ylabel('PC2')
        axs.set_title('Loading Plot of Top 2 Principal Components')

        

    def elbow_plot_regression(self, num_pcs_to_keep=None, target_slope=0.1):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if num_pcs_to_keep is None:
            num_pcs_to_keep = len(self.cum_var)
            
            
        # if self.cum_var is None or num_pcs_to_keep > len(self.cum_var):
        #     num_pcs_to_keep = len(self.cum_var)
        
        x = np.arange(1, num_pcs_to_keep+1)
        y = self.cum_var[:num_pcs_to_keep]
        
        
        plt.plot(x, y, marker='o', label='Cumulative Variance')
        
        #calc the differential slope
        slopes = np.diff(y) / np.diff(x)
        
        #find where the slope is closest to the target slope
        best_index = np.argmin(np.abs(slopes - target_slope))
        best_num_pcs = best_index + 1
            
        plt.axvline(best_num_pcs, linestyle='--', color='green', label=f'Best PCs = {best_num_pcs}')
        plt.xlabel('Number of Principle Components')
        plt.ylabel('Explained Variance in the model')
        plt.title("Elbow Plot")
        
        self.best_index = best_num_pcs
        return best_num_pcs
        
        

    def find_num_pcs_for_variance(self,explained_var, target_var=0.90):
        explained_var = np.real(explained_var)
        cumulative_variance = np.array(self.compute_cum_var(explained_var))
        
        #argmax takes the largest index corresponding to the PC at 90 we add 1 so that we can go from index to #
        num_pcs = np.argmax(cumulative_variance >= target_var) + 1  
        return num_pcs


