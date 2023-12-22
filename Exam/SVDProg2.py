import numpy as np
matrix = np.array([[5,6,7],[2,4,5],[9,8,7]])
u,s,vt = np.linalg.svd(matrix)
print(u)
print("Deependra")
print("",np.diag(s))
print("Deependra")
print(vt)

re = np.dot(u,np.dot(np.diag(s),vt))
print(re)