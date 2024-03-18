import numpy as np 
from scipy.linalg import lapack 
import matplotlib.pyplot as plt 

'''
frobenius norm of a matrix
'''
def norm(matrix):
  f = np.sqrt(np.sum(matrix**2))
  return f

''' 
read a file given the file name and 
return the data in the file as a tranposed numpy matrix
also returns rows m and columns n 
'''
def read_file(myFileName):  
  with open(myFileName, 'r') as myFile:
    #data = myFile.read()
    matrix = np.loadtxt(myFileName)
    matrix = matrix.T
    m, n = np.shape(matrix)
    return matrix, m, n
  
'''
returns 
U: left singular vectors matrix
s: array containing singular values
VT: transpose of right singular vectors matrix
info: 0 if it works, 1 else
''' 
def svd(matrix):
  U, s, VT, info = lapack.dgesvd(matrix)
  Sigma = np.diag(s)
  Sigma = np.zeros((len(U), len(VT)))
  Sigma_diags = np.diag(s)
  Sigma[:len(s), :len(s)] = Sigma_diags
  return U, Sigma, s, VT

'''
saves matrix as .dat file to 
base + filename
'''
def save_to_dat(matrix, base, filename):
  np.savetxt(base + filename, matrix, fmt = "%.16e", delimiter = "\t")
'''
takes in matrices and title of plot
'''
