import numpy as np

matrix = np.array([[5,6,4],
                   [2,5,6],
                   [3,5,6]])

U, S, VT = np.linalg.svd(matrix)

print("U Matrix")
print(U)
print(S)

print("S Diagonal Matrix")
print(np.diag(S))

print("VT Matrix")
print(VT)


re_matrix = np.dot(U, np.dot(np.diag(S), VT))
print(re_matrix)