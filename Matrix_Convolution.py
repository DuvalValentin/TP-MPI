import numpy as np
import math
import time

#needed for MPI treatment ------------
from mpi4py import MPI
comm = MPI.COMM_WORLD # to comunicate between nodes
rank= comm.Get_rank() # rank of one process
size= comm.Get_size() # number of proccess
# ------------------------------------
if(rank==0): #True si rank est 0, pas la peine de faire l'initialisation sur plusieurs noeuds
  start = time.time()
  
def printTable(A):
    for row in A:
      print(row)

# NxN matrix
N = 60

def getRowsStartEnd(node_number):
  row_start=int(node_number*N/size)#first ligne index
  row_end=int((node_number+1)*N/size)#last ligne index +1
  return(row_start,row_end)

if(rank==0):
  # Matrix for calculation input and output
  A = np.zeros((N-2, N-2))
  A = np.pad(A, pad_width=1, mode='constant', constant_values=1)
  printTable(A)
else:
  A=None #Initialisation in order to broadcast
A=comm.bcast(A,root=0)
# Sub_Matrix for calculation output temp
(row_start,row_end)=getRowsStartEnd(rank)
B = np.zeros((row_end-row_start,N))
converge = False
iteration_num = 0

while (converge==False):
  if(rank==0):
    iteration_num = iteration_num+1#seul le noeud 0 va compter les iterations
  diffnorm = 0.0
  # for convenience, use padding border
  A_padding = np.pad(A, pad_width=1, mode='constant', constant_values=0)#on ajoute des 0 tout autour#2 lors de la prod

  # partie a decouper -------------------------------------------------------------------------------
  for n in range (size):
    if(rank==n):
      (row_start,row_end)=getRowsStartEnd(rank)
      sub_copy_A_padding = A_padding[row_start:row_end+2]
      for i in range(0,row_end-row_start):# pour que ca corresponde au index de sub_copy_A_padding
        for j in range(N):
          # because we do padding, index changed
          #print("rang : "+str(rank)+" | i : "+str(i)+" | j : "+str(j))
          i_padd=i+1
          j_padd=j+1
          B[i][j] = 0.25*(sub_copy_A_padding[i_padd+1, j_padd]
                          + sub_copy_A_padding[i_padd-1, j_padd]
                          + sub_copy_A_padding[i_padd, j_padd+1]
                          + sub_copy_A_padding[i_padd, j_padd-1])
          diffnorm += math.sqrt((B[i, j] - A[i+row_start, j])*(B[i, j] - A[i+row_start, j]))
      comm.barrier()#on doit attendre tout les traitement avant de copier B dans A
  
  if(rank!=0):
    comm.send(B,0)
    comm.send(diffnorm,0)
  else:
    A[row_start:row_end]=B
    for node in range (1,size):
      (row_start,row_end)=getRowsStartEnd(node)
      A[row_start:row_end]=comm.recv(source=node)
      diffnorm=diffnorm+comm.recv(source=node)
    # check converge
    if diffnorm <= 0.01:
      converge = True
    else:
      converge = False
      
      
    
    if iteration_num % 100 == 0:
      print("*************")
      printTable(A)
      print("*************")
    
  A=comm.bcast(A,0)
  converge=comm.bcast(converge,0)
  comm.barrier()#on attends la nouvelle valeur de converge pour etre sur de ne pas faire un tour de boucle inutile

if (rank==0):
  print("*************")
  printTable(A)
  print("*************")
  print('Converge, iteration : %d' % iteration_num)
  print('Error : %f' % diffnorm)
  end = time.time()
  print('execution time : ')
  print(end - start)
  