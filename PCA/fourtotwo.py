import numpy as nm 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import pandas as pd 


df = pd.read_csv(filepath_or_buffer='input.data', 
    header=None, 
    sep=',')


tls.set_credentials_file(username='sahilhindwani', api_key='3wq407mwg5')


#STEP 1: Standardized the data
#standard_data = StandardScaler().fit_transform(data) #The data should be standardized as taken for all machine level programs



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

	print "Eigen values in descending order:"
	return eigen_pair



#STEP5: check which eigen values to take and which not to take

def get_cool_eigenvalues(eigenpairs):
	eigenvalues = [x[0] for x in eigenpairs]
	eigenvector = [x[1] for x in eigenpairs]

	total = sum(eigenvalues)
	perc =  [((eigenvalues[i]/total)*100) for i in range(len(eigenvalues))]

	for i in perc:
		print i
	x=2#raw_input("Enter the number of dimensions u want to reduce to based on variance percentage")
	return x



#STEP 6 : Now build the transformation matrix.

def calculate_transformation_matrix(eigenpairs):
	t_matrix = nm.hstack((eigenpairs[0][1].reshape(4,1),eigenpairs[1][1].reshape(4,1)))
	return t_matrix

#STEP 7: Now build the transformed data

def build_data(data,eigenpairs):
	return data.dot(calculate_transformation_matrix(eigenpairs))



'''Now complete code here'''

#data = nm.genfromtxt("input.data",delimiter = ",")
data = df.ix[:,0:3].values
y = df.ix[:,4].values
print data
mean1 = data.mean(axis = 0)
standard_data = StandardScaler().fit_transform(data)
print "standard_data:"
print standard_data


cov_matrix = calculate_cov(standard_data)
eigenvalues,eigenvector = calculate_eigen(cov_matrix)
eigenpair = sort_eigen_pairs(eigenvalues,eigenvector)
print 'eigen pairs are here!!'
print eigenpair
print '\n'

x=get_cool_eigenvalues(eigenpair)
t_matrix = calculate_transformation_matrix(eigenpair)
Y = build_data(standard_data,eigenpair)
traces = []



for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):

    trace = Scatter(
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)
data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
py.plot(fig)
print Y

print "Reconstruced Matrix:"
X = Y.dot(t_matrix.T)
print X 

print sum(sum(nm.abs(X-standard_data))+mean1)