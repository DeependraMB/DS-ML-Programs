import numpy as np

R = int(input("Enter the Number of rows for matrix1 ::"))
C = int(input("Enter the Number of columns for matrix 1::"))
matrix1=[]
print("Enter the entries::\n")
for i in range(R):
    a=[]
    for j in range(C):
        a.append(int(input()))
    matrix1.append(a)

for i in range(R):
    for j in range(C):
        print(matrix1[i][j],end="")
    print()



R1 = int(input("Enter the Number of rows for matrix2 ::"))
C1 = int(input("Enter the Number of columns for matrix 2::"))
matrix2=[]
print("Enter the entries::\n")
for i in range(R1):
    a2=[]
    for j in range(C1):
        a2.append(int(input()))
    matrix2.append(a2)

for i in range(R1):
    for j in range(C1):
        print(matrix2[i][j],end="")
    print()

sum = np.add(matrix1,matrix2)
print("Matrix Addition:\n")

for i in range(R):
    for j in range(C):
        print(sum[i][j],end="")
    print()


