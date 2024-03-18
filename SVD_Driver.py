import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


import SVD_functions
read_file = SVD_functions.read_file
svd = SVD_functions.svd
norm = SVD_functions.norm
save_to_dat = SVD_functions.save_to_dat


if __name__ == "__main__":
  base = "/Users/alexandranava/Desktop/AM213A/Final Project/"
  save_data = 1
  show_images = 1
  #-------------------------------------------------
  '''1. Read in Data - 
  matrix T is the tranposed data, 
  m is rows in transposed matrix 
  n is columns in transposed matrix
  m,n = 1920, 1279
  '''
  doggy = "dog_bw_data.dat"
  matrixT, m, n = read_file(doggy) 
  print("Reading in " + doggy + "...")
  #-------------------------------------------------
  '''2. SVD Decomposition - 
  ||A - U@Sigma@VT|| # -> 4.021234833774079e-09 <3
  '''
  U, Sigma, s, VT = svd(matrixT)
  U = U.copy()
  VT = VT.copy()
  Sigma = Sigma.copy()
  s = s.copy()
  #-------------------------------------------------
  '''3. Reconstructing Compressed Images:
  Sigma_k : reduced Sigma
  A_k : A as a reduced SVD
  Each reduced SVD is saved to base + "Image_appn_1xxxxx.dat"
  '''
  print("Computing reduced SVDs...")
  ks = [10, 20, 40, 80, 160, 320, 640, 1229]
  errors = []
  for k in ks:
    A_k = U[:, :k] @ Sigma[0:k, :k] @ VT[:k, :]
  #-------------------------------------------------
    '''4. Saving reduced SVDs to .dat files:
    '''
    tmp_id = ""
    for i in range(5 - len(str(k))):
      tmp_id += "0"
    tmp_id += str(k)
    filename = "Image_appn_1" + tmp_id + ".dat"
    if save_data == 1:
      save_to_dat(A_k, base, filename)
  #-------------------------------------------------
    '''5. Plotting compressed data
    '''
    if show_images == 1:
      A_k_uint = A_k.copy()
      A_k_uint8 = A_k_uint.astype(np.uint8)
      img = plt.imshow(A_k.T)
      img.set_cmap("grey")
      plt.axis("off")
      plt.title("Image Reconstructed with " + str(k) + " Singular Values")
      plt.savefig(tmp_id + ".jpg")
  #-------------------------------------------------
    '''6. Corresponding errors:
    '''
    
    E = norm(matrixT - A_k)/(n*m) #LAST ONE IS NOT 0? 
    errors.append([k, E])
  errors_df = pd.DataFrame(errors)
  errors_df.to_csv("errors.csv") 
  errors_df.plot(x = 0, y = 1, kind = "line", marker = 'o', color = "pink", label = "Error")
  plt.xlabel("Number of Singular Values")
  plt.ylabel("Error")
  plt.title("Error(Singular Values)")
  plt.savefig("errors.jpg")
  plt.close()

  '''produce original image'''
  img = plt.imshow(matrixT.T)
  img.set_cmap("grey")
  plt.axis("off")
  plt.title("Original Doggy")
  plt.savefig("puppy.jpg")

  print("First 10 singular values:")
  print(s[0:10])

  for k in ks:
    print("singular value for k = " + str(k))
    print(s[k-1])


#ref 
#https://www.youtube.com/watch?v=H7qMMudo3e8&t=305s

