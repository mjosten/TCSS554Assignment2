"""
Michael Josten
TCSS554 Information Retrieval 
Assginment 2:

This Assignment will implement the PageRank Algorithm for ranking documents (nodes)
based on the amount of in-links to a node and the value of each in-link. We will also implement a 
GoogleMatrix which will introduce a random teleport feature to make the matrix stochastic, aperiodic,
and irreducible which will solve the problems of spider-trap and deadend surfing. 

Program Input: text file which contains an adjacency matrix which describes a M matrix,
which i and j are nodes and the cell ij in M, contains a value k where k=1 if i -> j. i -> j denotes
that i links to j. k=0 if i doesn't link to j.

Beta (Dampening factor) = 0.85
Theta = 0.0001: this will be used for a convergence threshold.

Program Output: A rank vector after convergence and the number of iterations to converge rounded to 4 decimal
decimal places.

Homework Submision Content:
Output of Matrix M,
Output of Matrix A after applying teleportation,
Original Rank Vector
Converged Rank Vector for Matrix M and A.
Iterations to get a converged rank vector with Matrix M and A.
"""
import sys

BETA = 0.85
EPSILON = 0.000001

#Main function for the PageRank program.
def main():

    #Command line argument for file path
    arg = 'test1.txt'
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
    fInput = open(arg, 'r')

    #adjacency matrix
    adjMat = getAdjacencyMatrix(fInput) 
    #Matrix M
    M = createMatrixM(adjMat)
    # #Matrix A
    A = createMatrixA(M)
    # #Original Rank Vector
    R0 = createBaseRankVector(len(M))
    # # Converge the rank vectors
    convergedRankM, iterationsM = convergeRankVector(M, R0)
    convergedRankA, iterationsA = convergeRankVector(A, R0)

    # display outputs
    print("Matrix M")
    printMat(M)
    print("\n" + "Matrix A")
    printMat(A)
    print("\n" + "Original Rank Vector")
    printVect(R0)
    print("\n" + "Converged Rank Vector for Matrix M: ")
    printVect(convergedRankM)
    print("Iterations to get converged rank vector with Matrix M: ", end='')
    print(iterationsM)
    print("\n" + "Converged Rank Vector for Matrix A: ")
    printVect(convergedRankA)
    print("Iterations to get converged rank vector with Matrix A: ", end='')
    print(iterationsA)
    


# Function that will power iterate the rank vector based on the Matrix passed
# and will stop when the difference between |Rt+1 - Rt| < EPSILON
def convergeRankVector(M, R):
    newR = iterateRankVector(M, R)
    # set iterations count
    iterations = 0
    if (vectorDifference(R, newR) > 0):
        iterations = 1

    while (vectorDifference(newR, R) > EPSILON):
        R = newR
        newR = iterateRankVector(M, R)
        iterations += 1
    
    return newR, iterations
    
# Function that will iterate the rank vector based on the matrix,
# which is basically multiplying the rank vector by the matrix.
def iterateRankVector(M, R):
    newR = []
    for i in range(len(M)):
        tempSum = 0
        for j in range(len(M[i])):
            tempSum += M[i][j] * R[j]
        newR.append(tempSum)
    return newR

# Helper function to take the difference of vectors and sum up the differences 
# to a single value. Will also apply the absolute value to the difference.
def vectorDifference(V1, V2):
    dif = 0
    if (len(V1) == len(V2)):
        for c1, c2 in zip(V1, V2):
            dif += c1 - c2
    return abs(dif)

# Function that will create Matrix A Which implements random teleport,
# which will randomly teleport with probability(1-beta)
# Matrix A = beta * M + (1-beta) * (1/n * 1 * 1^t)
# where (1/n * 1 * 1^t) is a matrix the size of m with each cell containing 1/n of the cloumn.
def createMatrixA(M):
    distMat = createDistMatrix(M)
    A = []
    for i in range(len(M)):
        tempRow = []
        for j in range(len(M[i])):
            cellTotal = (BETA * M[i][j]) + ((1 - BETA) * distMat[i][j])
            tempRow.append(cellTotal)
        A.append(tempRow)
    return A

# Function that will create Matrix M
# if i -> j then M at ij = 1/d where d is the number of outlinks of i
# else M at ij = 0
def createMatrixM(adjMat):
    M = __copyMat__(adjMat)
    numbOfLinks = generateLinks(adjMat)
    for i in range(len(M[0])):
        links = numbOfLinks[i]
        for j in range(len(M)):
            if (M[j][i] == 1):
                M[j][i] = 1/links
    return M

# Function that will return a list of the sum of each column
# this is to identify the number of outlinks of i which is the column
def generateLinks(adjMat):
    result = []
    for i in range(len(adjMat[0])):
        columnSum = 0
        for j in range(len(adjMat)):
            columnSum += adjMat[j][i]
        result.append(columnSum)
    return result

# Function that will read graph.txt and return an adjacency matrix.
# 1. create a dictionary of {node# : [list of nodes pointed to by node]}
# 2. create a dictionary of {node# : # of outlinks of node}
# 3. using this information create the adjacency matrix.
def getAdjacencyMatrix(fGraph):

    input = []
    for line in fGraph:
        input.append(line.split())
    
    nodesDict = createNodeDicts(input)
    maxDictAmount = getMaxAmount(nodesDict)
    
    # Initialize the adjacency matrix
    adjMatrix = []
    for _ in range(maxDictAmount):
        tmpList = []
        for _ in range(maxDictAmount):
            tmpList.append(0)
        adjMatrix.append(tmpList)

    # populate the adjacency matrix with node link information
    for node, outNodes in nodesDict.items():
        for el in outNodes:
            if (node-1 >= 0 and el-1 >= 0):
                adjMatrix[el-1][node-1] = 1


    return adjMatrix

# Helper function that will create dictionaries of node out nodes and node # of outlinks per node
# input is the input matrix file put into a 2D list
def createNodeDicts(input):
    nodeOutNodesDict = {}
    for el in input:
        node = int(el[0])
        outNode = int(el[1])

        if node not in nodeOutNodesDict:
            nodeOutNodesDict[node] = [outNode]
        else:
            if outNode not in nodeOutNodesDict[node]:
                nodeOutNodesDict[node].append(outNode)

    return nodeOutNodesDict

# Helper function that will get the max value out of all the keys and values
def getMaxAmount(dct):
    maxValue = 0
    for k, v in dct.items():
        maxValue = max(maxValue, k)
        for el in v:
            maxValue = max(maxValue, el)
    
    return maxValue


# Helper function that will create a distribution matrix where
# each cell contains 1/n where 1/n is the total number of cells in a column.
# used for calculating Matrix A
def createDistMatrix(M):
    distMat = []
    for i in range(len(M)):
        d = len(M[i])
        #tempRow for each row of distMat
        tempRow = []
        for _ in range(d):
            tempRow.append(1/d)
        distMat.append(tempRow)
    return distMat

# Helper function that will create a vector with values evenly 
# distributed based on the number of nodes for the PageRank algorithm.
# R0 = [1/n, 1/n, ..., 1/n]^T where T is the number of nodes.
def createBaseRankVector(nNodes):
    R0 = []
    for _ in range(nNodes):
        R0.append(1/nNodes)
    return R0


#Helper function that prints a matrix to the console
def printMat(M):
    for i in range(len(M)):
        print("[", end='')
        for j in range(len(M[i])):
            print("{:.5f}".format(round(M[i][j], 5)), end='')
            #print("{:.8f}".format(round(M[i][j], 8)), end='')
            if j != len(M[i])-1:
                print(", ", end="")

        print("]")
    return

# Helper function that prints a vector to the console.
def printVect(V):
    print('[', end='')
    for i in range(len(V)):
        print("{:.5}".format(round(V[i], 5)), end='')
        #print("{:.8f}".format(round(V[i], 8)), end='')
        if i != len(V) - 1:
            print(", ", end='')
    print(']')
    return

def printDict(dct):
    for k, v in dct.items():
        print("{} : {}".format(k, v))

# Helper function that performs a deep copy of a 2d matrix
def __copyMat__(M):
    newM = []
    for i in range(len(M)):
        newM.append([])
        for j in range(len(M[0])):
            newM[i].append(M[i][j])
    return newM

if __name__ == "__main__":
    main()