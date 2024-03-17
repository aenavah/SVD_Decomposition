import pandas as pd
import numpy as np

import SVD_functions
read_file = SVD_functions.read_file
svd = SVD_functions.svd
norm = SVD_functions.norm
reduced_sigma = SVD_functions.reduced_sigma
save_to_dat = SVD_functions.save_to_dat
plot_compressed = SVD_functions.plot_compressed


if __name__ == "__main__":
  base = "/Users/alexandranava/Desktop/AM213A/Final Project/"
  #-------------------------------------------------
  '''1
  Read in Data - 
  matrix T is the tranposed data, 
  m is rows in transposed matrix 
  n is columns in transposed matrix
  m,n = 1920, 1279
  '''
  doggy = "dog_bw_data.dat"
  matrixT, m, n = read_file(doggy) 
  print("Reading in " + doggy + "...")
  #-------------------------------------------------
  '''2
  SVD Decomposition - 
  ||A - U@Sigma@VT|| # -> 4.021234833774079e-09
  '''
  U, Sigma, VT = svd(matrixT)
  U = U.copy()
  VT = VT.copy()
  Sigma = Sigma.copy()
  #-------------------------------------------------
  '''3 and 4
  Reconstructing Compressed Images:
  Sigma_k : reduced Sigma
  A_k : A as a reduced SVD

  Each reduced SVD is saved to base + "Image_appn_1xxxxx.dat"
  '''
  print("Computing reduced SVDs...")
  ks = [10, 20, 40, 80, 160, 320, 640, 1229]
  files = []
  for k in ks:
    Sigma_k = reduced_sigma(Sigma, k)
    A_k = U @ Sigma_k @ VT
    #creating naming
    tmp_id = ""
    for i in range(5 - len(str(k))):
      tmp_id += "0"
    tmp_id += str(k)
    filename = "Image_appn_1" + tmp_id + ".dat"
    files.append(filename)
    #saving compressed A
    print("Saved reduced SVD with k = " + str(k) + " to " + base + filename + "...")
    save_to_dat(A_k, base, filename)

  '''
  5 Plotting compressed data
  '''
  for file in files:
    compressed_data, m_k, n_k = read_file(file)
    title = file.replace("Image_appn_1", '')
    title = title.replace(".dat", "")
    while title[0] == "0":
      title = title[1:]
    k = title
    plot_compressed(compressed_data, title)


