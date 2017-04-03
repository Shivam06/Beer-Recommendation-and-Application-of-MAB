# Changes to add on are specification of number of recommendations you want for a particular customer (Done)
# Add documentation for the code (Done)
# Add visualizations if possible (Done)
# Correct product = "jac" and "dist" case

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from metrics import *

class Recommendation:
    def __init__(self, utility_matrix, algo = "col", type = "item" ,product = "cos", K = 10):
        """
        Recommendation module - 
        Parameters for Recommendation class object : 1) utility_matrix, 2) algo - "col" for collaborative filtering or "knn" for 
        K-Nearest Neighbor, 3) type - "item" for item based, "user" for user based recommendations, 4) product - "cos" for cossine                 similarity, "jac" for jaccard similarity, "dist" for euclidean distance in case of knn, 5) K - number of neighbors in case of KNN.
        
        """
        self.type = type
        self.algo = algo
        self.K = K
        self.product =  product
        self.similarity_matrix = np.array([])
        self.utility_matrix = np.array(utility_matrix)
        self.final_score_matrix = np.zeros(self.utility_matrix.shape) 

    
    def similarity_mat(self):
        """
        Parameters : None
        Output : Returns the similarity matrix for utility matrix inputted to Recommendation object.
        
        """
        if self.type == "item":
            self.similarity_matrix = np.identity(self.utility_matrix.shape[1]) # item X item
            for i in range(self.similarity_matrix.shape[0]):
                for j in range(i+1,self.similarity_matrix.shape[1]):
                    self.similarity_matrix[i,j] = product(self.utility_matrix[:,i],self.utility_matrix[:,j], self.product)
            self.similarity_matrix = self.similarity_matrix + self.similarity_matrix.T - np.identity(self.utility_matrix.shape[1])
            
        elif self.type == "user":
            self.similarity_matrix = np.identity(self.utility_matrix.shape[0]) # user X user
            for i in range(self.similarity_matrix.shape[0]):
                for j in range(i+1,self.similarity_matrix.shape[1]):
                    self.similarity_matrix[i,j] = product(self.utility_matrix[i,:], self.utility_matrix[j,:], self.product)
            self.similarity_matrix = self.similarity_matrix + self.similarity_matrix.T - np.identity(self.utility_matrix.shape[0])
        return self.similarity_matrix
    
    def final_recommendation_score_matrix (self):
        """
        Parameters: None,
        Ouput: score_matrix representing score of each item (columns) for every users (rows).
        
        * NOTE: Higher score values for an item is better for a particular user (row) in case algo is "col". Else, Lower score is better in 
        case of KNN.
        
        """
        if self.algo == "col":
            if self.type == "item":
                self.final_score_matrix = np.dot(self.utility_matrix,self.similarity_matrix)
            elif self.type == "user":
                self.final_score_matrix = np.dot(self.similarity_matrix,self.utility_matrix)
        elif self.algo == "knn":
            self.modified_similarity_matrix = self.similarity_matrix - 2*np.identity(self.similarity_matrix.shape[0])
            value_matrix = np.zeros((self.K,self.similarity_matrix.shape[1]))
            index_matrix = np.zeros((self.K,self.similarity_matrix.shape[1]))
            for i in range(self.similarity_matrix.shape[1]):
                temp = self.modified_similarity_matrix[:,i]
                ind = np.argpartition(temp, -self.K)[-self.K:]
                index_matrix[:,i] = ind
                value_matrix[:,i] = temp[ind] 
            self.final_recommendation_score_matrix_helper_for_different_knn_types(self.type, index_matrix, value_matrix)   
        return self.final_score_matrix
    

    def final_recommendation_score_matrix_helper_for_different_knn_types(self, type_of_method, index_matrix, value_matrix):
        """ 
        Helper function for final_recommendation_score_matrix method.
        
        Parameters - 1) type_of_method - "user" for user based and "item" for item based recommendations, 2) index_matrix, 3) value_matrix.
        Output: None
        
        """
        if type_of_method == "user":
            for i in range(index_matrix.shape[1]):
                temp = index_matrix[:,i].astype(int)
                k_similar_users_profiles = self.utility_matrix[temp]
                k_similar_users_similarity_values = value_matrix[:,i]
                self.final_score_matrix[i,:] = np.dot(k_similar_users_similarity_values,k_similar_users_profiles)
        elif type_of_method == "item":
            for i in range(index_matrix.shape[1]):
                temp = index_matrix[:,i].astype(int)
                k_similar_item_profiles = self.utility_matrix.T[temp]
                k_similar_item_similarity_values = value_matrix[:,i]
                self.final_score_matrix[:,i] = np.dot(k_similar_item_similarity_values,k_similar_item_profiles)
    
    def normalized_final_recommendation_score_matrix(self):
        self.normalized_final_score_matrix = np.array([a/np.sum(a) for a in self.final_score_matrix if np.sum(a) != 0])
        return self.normalized_final_score_matrix
    
    def recommendation_ranking(self):
        """
        Parameters - 1) iu - final recommendation score matrix (I X U).
        Ouput - 1) Recommendation Ranking matrix.
        
        """
        iu = self.final_recommendation_score_matrix()
        new_iu = []
        for row in iu:
            li = []
            temp = row
            if self.product != "dist":
                temp = -np.sort(-temp)
                for element in row:
                    li.append(binary_search_opp(temp,element)+1)             
            else:
                temp = np.sort(temp)
                for element in row:
                    li.append(np.searchsorted(temp,element)+1)
            new_iu.append(li)
        return np.array(new_iu) 

