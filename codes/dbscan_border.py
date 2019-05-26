import numpy as numpy
import scipy as scipy
from sklearn import cluster
import matplotlib.pyplot as plt
 
 
 
def set2List(NumpyArray):
    list = []
    for item in NumpyArray:
        list.append(item.tolist())
    return list
 
 
# def GenerateData():
#     x1=numpy.random.randn(50  2)+12
#     # x2x=numpy.random.randn(80  1)+12
#     # x2y=numpy.random.randn(80  1)
#     # x2=numpy.column_stack((x2x  x2y))
#     # x3=numpy.random.randn(100  2)+8
#     # x4=numpy.random.randn(120  2)+15
#     # z=numpy.concatenate((x1  x2  x3  x4))
#     return x1
 
 
def DBSCAN(Dataset,Epsilon_min,Epsilon_max,MinumumPoints,DistanceMethod = 'euclidean'):
#    Dataset is a mxn matrix   m is number of item and n is the dimension of data
    m ,n=Dataset.shape
    Visited=numpy.zeros(m,'int')
    Type=numpy.zeros(m)
    Membership_value = numpy.zeros(m)
#   -1 noise   outlier
#    0 border
#    1 core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=numpy.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset,DistanceMethod))
    # print(DistanceMatrix)
    Border_membership = []
    for i in range(m):
        Border_membership.append([0,0,0,0])
    for i in range(m):
        if Visited[i]==0:
            Visited[i]=1
            PointNeighbors=numpy.where(DistanceMatrix[i]<=Epsilon_min)[0]
            # print(PointNeighbors)
            if len(PointNeighbors)<MinumumPoints and PointClusterNumber[i] == 0:
                Type[i] = -1
            elif len(PointNeighbors)>=MinumumPoints:
                Type[i] = 1
                for k in range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                
                PointNeighbors=set2List(PointNeighbors) 

                ExpandClsuter(Dataset[i],i,Membership_value,Border_membership,PointNeighbors,Cluster,MinumumPoints,Epsilon_min,Epsilon_max,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex,Type )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                 
    
    return PointClusterNumber,Border_membership,Type
 
 
 
def ExpandClsuter(PointToExapnd,core_point,Membership_value,Border_membership,PointNeighbors,Cluster,MinumumPoints,Epsilon_min,Epsilon_max , Visited , DistanceMatrix, PointClusterNumber,PointClusterNumberIndex,Type  ):
    # print("ExpandClsuter")
    FuzzyBorderpts11=[]
    FuzzyBorderpts12=[]
    FuzzyBorderpts=[]
    FuzzyBorderpts21=[]
    FuzzyBorderpts22=[]
    FuzzyBorderpts2=[]
    FuzzyBorderpts11 = numpy.where((DistanceMatrix[core_point]<=Epsilon_max))[0]
    FuzzyBorderpts12 = numpy.where(DistanceMatrix[core_point]>Epsilon_min)[0]
    FuzzyBorderpts = numpy.intersect1d(FuzzyBorderpts11,FuzzyBorderpts12)
    FuzzyBorderpts = set2List(FuzzyBorderpts)
    Neighbors=[]
    for i in PointNeighbors:
        if Visited[i]==0:
            Visited[i]=1
            Neighbors=numpy.where(DistanceMatrix[i]<=Epsilon_min)[0]
            # print(Neighbors)
            if len(Neighbors)>=MinumumPoints:
#                Neighbors merge with PointNeighbors
                Type[i] = 1
                for j in Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except ValueError:
                        PointNeighbors.append(j)
                FuzzyBorderpts21 = numpy.where(DistanceMatrix[i]<=Epsilon_max)[0]
                FuzzyBorderpts22 = numpy.where(DistanceMatrix[i]>Epsilon_min)[0]
                FuzzyBorderpts2 = numpy.intersect1d(FuzzyBorderpts21,FuzzyBorderpts22)

                for j in FuzzyBorderpts2:
                    try:
                        FuzzyBorderpts.index(j)
                    except ValueError:
                        FuzzyBorderpts.append(j)
                if PointClusterNumber[i]==0:
                    Cluster.append(i)
                    PointClusterNumber[i]=PointClusterNumberIndex  
            
            else:
                FuzzyBorderpts.append(i)
                

    # print("FuzzyBorderpts:    "  FuzzyBorderpts)
    for i in FuzzyBorderpts:
        if i not in Cluster:
            Type[i] = int(0) 
        minimum = 999
        index = -1
        for j in Cluster:
            if minimum > DistanceMatrix[i][j]:
                minimum = DistanceMatrix[i][j]
                index1 = j
        PointClusterNumber[i] = PointClusterNumberIndex
        if DistanceMatrix[i][index1]<=Epsilon_min:
            Border_membership[i][PointClusterNumberIndex]=1
        elif  DistanceMatrix[i][index1]>Epsilon_min and DistanceMatrix[i][index1]<=Epsilon_max:
            print(PointClusterNumberIndex  i)
            Border_membership[i][PointClusterNumberIndex]=((Epsilon_max-DistanceMatrix[i][index1])/(Epsilon_max-Epsilon_min))
        else:
            Border_membership[i][PointClusterNumberIndex] = 0    
    # print("return to DBSCAN") 
    return 
 
#Generating some data with normal distribution at 
#(0  0)
#(8  8)
#(12  0)
#(15  15)

# Data=GenerateData()

X = numpy.loadtxt("glass.txt" , dtype = 'int')
Data = X[:,1:-1]
# Data = numpy.array([[1  1]  [2  1]  [3  1]  [2  2]  [3  2]  [2  3]  [7  1]  [8  1]  [9  1][7  2]  [8  2]  [7  7]])
#Adding some noise with uniform distribution 
#X between [-3  17]  
#Y between [-3  17]
# noise=scipy.rand(50  2)*20 -3
 
# Noisy_Data=numpy.concatenate((Data  noise))
size=20
 
 
# fig = plt.figure()
# ax1=fig.add_subplot(2  1  1) #row   column   figure number
# ax2 = fig.add_subplot(212)
 
# ax1.scatter(Data[:  0]  Data[:  1]   alpha =  0.5 )
# ax1.scatter(noise[:  0]  noise[:  1]  color='red'   alpha =  0.5)
# ax2.scatter(noise[:  0]  noise[:  1]  color='red'   alpha =  0.5)
 
 
Epsilon_min=1
Epsilon_max=1.5
MinumumPoints=5
result,Border_membership,Type = DBSCAN(Data,Epsilon_min,Epsilon_max,MinumumPoints)
 
#printed numbers are cluster numbers
print (result,"\n",Type  "\n")
for i in range(len(Border_membership)):
    print(Border_membership[i]) 

#print "Noisy_Data"
#print Noisy_Data.shape
#print Noisy_Data
 
# for i in range(len(result)):
    # ax2.scatter(Noisy_Data[i][0],Noisy_Data[i][1],color='yellow',alpha =  0.5)
      
plt.show()