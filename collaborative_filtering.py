# This is collaborative filtering module. It takes as input - path of the data in excel format, type specifying either "item" or "user", product specifying the similarity product taking as input either "cos" for cossine and "jac" for jaccardian product and K specifyin K neighbors in case algo is K-Nearest Neighbor and input being "knn" and algo takes as input either "col" for collaborative filtering and "knn" for KNN.

# Changes to add on are specification of number of recommendations you want for a particular customer 
# Add documentation for the code
# Add visualizations if possible
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from metrics import *

class collaborative_filtering:
    def __init__(self, utility_matrix, type = "item" ,product = "cos", algo = "col", K = 10):
        self.type = type
        self.algo = algo
        self.K = K
        self.product =  product
        self.similarity_matrix = np.array([])
        self.utility_matrix = np.array(utility_matrix)  # users X items
        self.final_score_matrix = np.zeros(self.utility_matrix.shape)
        
    def similarity_mat(self):
        
        if self.type == "item":
            self.similarity_matrix = np.identity(self.utility_matrix.shape[1]) # item X item
            for i in range(self.similarity_matrix.shape[0]):
                for j in range(i,self.similarity_matrix.shape[1]):
                    if self.algo == "knn":
                        self.similarity_matrix[i,j] = distance(self.utility_matrix[:,i],self.utility_matrix[:,j])
                    elif self.product == "cos":
                        self.similarity_matrix[i,j] = cossine_product(self.utility_matrix[:,i],self.utility_matrix[:,j])
                    elif self.product == "jac":
                        self.similarity_matrix[i,j] = jaccardian_product(self.utility_matrix[:,i],self.utility_matrix[:,j])
            

        elif self.type == "user":
            self.similarity_matrix = np.identity(self.utility_matrix.shape[0]) # user X user
            for i in range(self.similarity_matrix.shape[0]):
                for j in range(i,self.similarity_matrix.shape[1]):
                    if self.algo == "knn":
                        self.similarity_matrix[i,j] = distance(self.utility_matrix[i,:],self.utility_matrix[j,:])
                    elif self.product == "cos":
                        self.similarity_matrix[i,j] = cossine_product(self.utility_matrix[i,:],self.utility_matrix[j,:])
                    elif self.product == "jac":
                        self.similarity_matrix[i,j] = jaccardian_product(self.utility_matrix[i,:],self.utility_matrix[j,:])
        
        if self.algo == "col":
            self.similarity_matrix = self.similarity_matrix + self.similarity_matrix.T - np.identity(self.similarity_matrix.shape[0])
        else:
            self.similarity_matrix = self.similarity_matrix + self.similarity_matrix.T 
        #self.similarity_matrix = self.similarity_matrix - np.identity(self.similarity_matrix.shape[0])
        
    def final_recommendation_score_matrix (self):
        if self.algo == "col":
            if self.type == "item":
                self.final_score_matrix = np.dot(self.utility_matrix,self.similarity_matrix)
            elif self.type == "user":
                self.final_score_matrix = np.dot(self.similarity_matrix,self.utility_matrix)
        
        elif self.algo == "knn":
            self.modified_similarity_matrix = self.similarity_matrix - np.identity(self.similarity_matrix.shape[0])
            value_matrix = np.zeros((self.K,self.similarity_matrix.shape[1]))
            index_matrix = np.zeros((self.K,self.similarity_matrix.shape[1]))
            for i in range(self.similarity_matrix.shape[1]):
                temp = self.modified_similarity_matrix[:,i]
                ind = np.argpartition(temp, -self.K)[-self.K:]
                index_matrix[:,i] = ind
                value_matrix[:,i] = temp[ind] 
            if self.type == "user":
                for i in range(index_matrix.shape[1]):
                    temp = index_matrix[:,i].astype(int)           ## Reduce Complexity
                    k_similar_users_profiles = self.utility_matrix[temp]
                    k_similar_users_similarity_values = value_matrix[:,i]
                    self.final_score_matrix[i,:] = np.dot(k_similar_users_similarity_values,k_similar_users_profiles)
                    
            elif self.type == "item":
                for i in range(index_matrix.shape[1]):
                    temp = index_matrix[:,i].astype(int)           ## Reduce Complexity
                    k_similar_item_profiles = self.utility_matrix.T[temp]
                    k_similar_item_similarity_values = value_matrix[:,i]
                    self.final_score_matrix[:,i] = np.dot(k_similar_item_similarity_values,k_similar_item_profiles)
        return self.final_score_matrix
                
    def recommendation_ranking(self,iu):
        new_iu = []
        for row in iu:
            li = []
            temp = row
            temp = -np.sort(-temp)
            for element in row:
                li.append(binary_search_opp(temp,element)+1)
            new_iu.append(li)
        return np.array(new_iu)
    
    def output_recommendations(self,col1_name = "Customer_id", col2_name = "Product_id", hash1 = {}, hash2 = {}):
        self.similarity_mat()
        self.final_recommendation_score_matrix()
        ranking_matrix = self.recommendation_ranking(self.final_score_matrix)
        output = np.ndarray((self.utility_matrix.size,5),dtype = np.object)
        idx = 0
        
        if len(hash1.keys()) == 0:
            hash1 = {i:i for i in range(self.utility_matrix.shape[0])}
        if len(hash2.keys()) == 0:
            hash2 = {i:i for i in range(self.utility_matrix.shape[1])}
            
        for i in range(self.utility_matrix.shape[0]):
            for j in range(self.utility_matrix.shape[1]):
                output[idx][0] = hash1[i]
                output[idx][1] = hash2[j]
                output[idx][2] = self.final_score_matrix[i,j]
                output[idx][3] = ranking_matrix[i,j]
                output[idx][4] = int(self.utility_matrix[i,j] > 0)
                idx += 1
                
        output_df = pd.DataFrame(output)
        output_df.columns = ["Customer_id","Product_id","Recommended_score","Rank","Flag"]
        output_df = output_df.sort_values(by = ["Customer_id","Rank"])
        output_df = pd.DataFrame(output_df.values)
        output_df.columns = [col1_name, col2_name, "Recommended_score", "Rank", "Flag"]
        return output_df
    

def binary_search_opp(arr, num):
    min = 0
    max = len(arr) - 1
    if arr[min] == num:
        return min
    if arr[max] == num:
        return max
    while (min < max):
        mid = min + int((max - min)/2)
        if arr[mid] == num:
            return mid
        elif arr[mid] > num:
            min = mid 
        else:
            max = mid