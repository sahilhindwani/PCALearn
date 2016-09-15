import numpy as nm 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data  = nm.genfromtxt('data.dat',delimiter = ',')
print data


#STEP 1: Standardized the data
standard_data = StandardScaler().fit_transform(data) #The data should be standardized as taken for all machine level programs



'''
centered_data = data-data.mean(axis = 0) # This will also give u the standardized data in this what we are doing is subtracting the array from their mean.
if centered_data.all() == standard_data.all():
	print "YES"'''


#STEP 2:find covariance matrix
def calculate_cov(data):
	return (standard_data - standard_data.mean(axis = 0)).T.dot((standard_data - standard_data.mean(axis = 0)))


#STEP3 : find eigen values and eigen vectors
def calculate_eigen(cov_matrix):
	return nm.linalg.eig(cov_matrix)
'''
	The linalg is the linear alzebra library of numpy therefore it can be used to calculate eigen value.
	The linalg library uses an algorithm to calculate eigen values which cannot work for large dimensional matrix
'''

#STEP4 : sort the eigen values according to their variances or in short according to its values.

def sort_eigen_pairs(eigen_values,eigenvector):
	eigen_pair=[(nm.abs(eigen_values[i]),eigenvector[:,i]) for i  in range(len(eigen_values))]
	eigen_pair.sort()
	eigen_pair.reverse()

	
	return eigen_pair



#STEP5: check which eigen values to take and which not to take

def get_cool_eigenvalues(eigenpairs):
	eigenvalues = eigenpairs[0]


	total = sum(eigenvalues)
	perc =  [((eigenvalues[i]/total)*100) for i in range(len(eigenvalues))]

	for i in perc:
		print i
	x=int(raw_input("Enter the number of dimensions u want to reduce to based on variance percentage"))
	return x



#STEP 6 : Now build the transformation matrix.

def calculate_transformation_matrix(eigenpairs):
	t_matrix = nm.hstack((eigenpairs[0][1].reshape(10,1),eigenpairs[1][1].reshape(10,1),eigenpairs[2][1].reshape(10,1)))
	return t_matrix

#STEP 7: Now build the transformed data

def build_data(data,eigenpairs):
	return data.dot(calculate_transformation_matrix(eigenpairs))



'''Now complete code here'''

data = nm.genfromtxt("data.dat",delimiter = ",")



standard_data = StandardScaler().fit_transform(data)
print "standard_data:"
print standard_data


cov_matrix = calculate_cov(standard_data)
eigenvalues,eigenvector = calculate_eigen(cov_matrix)
eigenpair = sort_eigen_pairs(eigenvalues,eigenvector)
t_matrix = calculate_transformation_matrix(eigenpair)
j = get_cool_eigenvalues(eigenpair)
Y = build_data(standard_data,eigenpair)
print Y