import SVD_functions
read_file = SVD_functions.read_file
svd = SVD_functions.svd
norm = SVD_functions.norm
reduced_sigma = SVD_functions.reduced_sigma
save_to_dat = SVD_functions.save_to_dat
plot_compressed = SVD_functions.plot_compressed

matrix = [[4, 0],[3, -5]]
U, Sigma, VT = svd(matrix)
print(U)
print(Sigma)
print(VT.T)