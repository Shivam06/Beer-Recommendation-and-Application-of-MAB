import numpy as np


def cossine_product(a,b):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 0
    else:
        return np.dot(a,b)/np.sqrt(np.sum(a**2)*np.sum(b**2))

def jaccardian_product(a,b):
    sum_and = 0
    sum_or = 0
    for i in range(len(a)):
        sum_and += int(int(np.ceil(a[i])) & int(np.ceil(b[i])))
        sum_or += int(np.ceil(a[i])) | int(np.ceil(b[i]))
    if sum_or == 0:
        return 0
    else:
        return float(sum_and)/float(sum_or)
    
def distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
    
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
            
def product(a, b, type_of_product):
    if type_of_product == "cos":
        return cossine_product(a,b)
    elif type_of_product == "jac":
        return jaccardian_product(a,b)
    elif type_of_product == "dist":
        return distance(a,b)