# Angelina Alderman
# COP 4500
# Assignment 3b

import numpy as np

# Solving system of linear equations

A = np.array(([2,-1,1,6],
	[1,3,1,0],
	[-1,5,4,-3])).astype(float)

def solve_matrix(A, reduced = False, augmented = False):

	rows, columns = A.shape
	offset = 1 if augmented else 0
	columns = columns - offset
	
	# Echelon
	for j in range(columns):
		A[j] = A[j] / A[j][j]
		for i in range(rows - j - 1):
			A[j + i + 1] = A[j + i +1] - A[j + i + 1][j] * A[j]
	if not reduced:
		return A

	# Reduced Echelon

	for j in range(columns):
		current_column = columns - j - 1
		for i in range(rows - j - 1):
			current_row = rows - j - 1
			altering_row = current_row - i - 1
			A[altering_row] = A[altering_row] - A[altering_row][current_column] * A[current_row]
	return A

# LU Factorization

B = np.array(([1,1,0,3],
	[2,1,-1,1],
	[3,-1,-1,2],
	[-1,2,3,-1])).astype(float)

determinant = np.linalg.det(B)

def LU(B):
	U = B.astype(float)
	dim = B.shape[0]
	L = np.eye(dim) # Identity matrix

	for j in range(dim):
		for i in range(dim - j - 1):
			if U[j][j] == 0:
				k = j + 1
				while k < dim:
					if U[j][k] != 0:
						U[[j,k]] = U[[k,j]]

						for n in range(j):
							L[[j,k], [n,n]] = L[[k,j], [n,n]]

							break
						k += 1

			multiplier = U[j + i + 1][j] / U[j][j]

			U[j+ i + 1] = U[j + i + 1] - multiplier * U[j]
			L[j + i + 1][j] = L[j + i + 1][j] + multiplier * L[j][j]

	return(L, U)

# Diagonally Dominant

C = np.array(([9,0,5,2,1],
	[3,9,1,2,1],
	[0,1,7,2,3],
	[4,2,3,12,2],
	[3,2,4,0,8]))

def diag_dom(C):
    D = np.diag(np.abs(C)) 
    S = np.sum(np.abs(C), axis=1) - D 
    if np.all(D > S):
        print("The matrix is diagonally dominant")
    else:
        print("The matrix is NOT diagonally dominant")
    return

# Positive Definite

E = np.array(([2,2,1],
	[2,3,0],
	[1,0,2]))

def pos_definite(E):
	if np.all(np.linalg.eigvals(E) > 0) and np.array_equal(E,E.T):
		print("The matrix is a positive definite matrix")
	else:
		print("The matrix is NOT a positive definite matrix")
	return

def main():

	print("Reduced echelon augmented matrix:\n", 
		solve_matrix(A, reduced = True, augmented = True))

	print("")
	print("Determinant =", determinant)
	print("")
	L, U = LU(B)
	print("L Matrix:\n", L)
	print("")
	print("U Matrix\n", U)
	print("")
	diag_dom(C)
	print("")
	pos_definite(E)
	print("")

if __name__ == "__main__":
	main()
