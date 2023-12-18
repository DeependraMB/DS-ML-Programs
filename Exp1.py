import numpy as np

R=int(input("Enter the number of rows for Matrix 1::"))
C=int(input("Enter the number of colums for Matrix 1::"))

matrix1=[]

print("Enter the entries::")

for i in range(R):
    a = []
    for j in range(C):
        a.append(int(input()))
    matrix1.append(a)

for i in range(R):
    for j in range(C):
        print(matrix1[i][j], end = " ")
    print()

R1=int(input("Enter the number of rows for Matrix 2::"))
C1=int(input("Enter the number of colums for Matrix 2::"))

matrix2=[]

print("Enter the entries::")


for i in range(R1):
    a = []
    for j in range(C1):
        a.append(int(input()))
    matrix2.append(a)

for i in range(R1):
    for j in range(C1):
        print(matrix2[i][j], end = " ")
    print()


sum=np.add(matrix1,matrix2)

print("Matrix Addition")

for i in range(R):
    for j in range(C):
        print(sum[i][j], end = " ")
    print()

sub=np.subtract(matrix1,matrix2)

print("Matrix Substraction")

for i in range(R):
    for j in range(C):
        print(sub[i][j], end = " ")
    print()

div=np.divide(matrix1,matrix2)

print("Matrix Division")

for i in range(R):
    for j in range(C):
        print(div[i][j], end = " ")
    print()

sqr=np.square(matrix1)

print("Matrix Square")

for i in range(R):
    for j in range(C):
        print(sqr[i][j], end = " ")
    print()