import numpy as np
import math
import time

start = time.time()

def printTable(A):
    for row in A:
      print(row)

# NxN matrix
MAX_N = 60

# Matrix for calculation input and output
A = np.zeros((MAX_N-2, MAX_N-2))
A = np.pad(A, pad_width=1, mode='constant', constant_values=1)

printTable(A)

# Matrix for calculation output temp
(row_num, col_num) = A.shape
B = np.zeros((row_num, col_num))

converge = False
iteration_num = 0
while (converge == False):
    iteration_num = iteration_num+1
    diffnorm = 0.0

    # for convenience, use padding border
    A_padding = np.pad(A, pad_width=1, mode='constant', constant_values=0)

    for i in range(row_num):
        for j in range(col_num):
            # because we do padding, index changed
            idx_i_A = i + 1
            idx_j_A = j + 1
            B[i][j] = 0.25*(A_padding[idx_i_A+1, idx_j_A]
                            + A_padding[idx_i_A-1, idx_j_A]
                            + A_padding[idx_i_A, idx_j_A+1]
                            + A_padding[idx_i_A, idx_j_A-1])
            # simple converge test
            diffnorm += math.sqrt((B[i, j] - A[i, j])*(B[i, j] - A[i, j]))
    A = np.copy(B)

    if iteration_num % 100 == 0:
        printTable(A)

    # check converge
    if diffnorm <= 0.01:
        print('Converge, iteration : %d' % iteration_num)
        print('Error : %f' % diffnorm)
        converge = True

printTable(A)
end = time.time()
print('execution time : ')
print(end - start)