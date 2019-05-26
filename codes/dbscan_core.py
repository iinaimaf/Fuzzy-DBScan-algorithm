import  numpy  as  numpy
import  scipy  as  scipy
from  sklearn  import  cluster
import  matplotlib.pyplot  as  plt

  
  
  
def  set2List(NumpyArray):
    list  =  []
    for  item  in  NumpyArray:
        list.append(item.tolist())
    return  list
  
  
def  GenerateData():
    x1=numpy.random.randn(5  2)
    x2x=numpy.random.randn(8  1)+2
    x2y=numpy.random.randn(8  1)
    x2=numpy.column_stack((x2x  x2y))
    x3=numpy.random.randn(10  2)+5
    x4=numpy.random.randn(12  2)+1
    z=numpy.concatenate((x1  x2  x3  x4))
    return  z
  
  
def  DBSCAN(Dataset, Epsilon ,MinumumPoints,  Max_value , DistanceMethod  =  'euclidean'):
#    Dataset  is  a  mxn  matrix    m  is  number  of  item  and  n  is  the  dimension  of  data
    m  n=Dataset.shape
    Visited=numpy.zeros(m  'int')
    Type=numpy.zeros(m)
    Membership_value  =  numpy.zeros(m)
#      -1  noise    outlier
#    0  border
#    1  core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=numpy.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix  =  scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset ,DistanceMethod))
    for  i  in  range(m):
        if  Visited[i]==0:
            Visited[i]=1
            PointNeighbors=numpy.where(DistanceMatrix[i]<Epsilon)[0]
            if  len(PointNeighbors)<MinumumPoints:
                Type[i]=-1
            else:
                if  len(PointNeighbors)>=Max_value:
                    Membership_value[i]  =  1
                    
                elif  len(PointNeighbors)  <  Max_value  and  len(PointNeighbors)  >  MinumumPoints:
                    Membership_value[i]  =  (len(PointNeighbors)  -  MinumumPoints)/(Max_value  -  MinumumPoints)
                      
                Type[i]  =  1
                for  k  in  range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                
                PointNeighbors=set2List(PointNeighbors)      
                ExpandClsuter(Dataset[i], Membership_value ,   PointNeighbors , Cluster , MinumumPoints , Epsilon , Visited , DistanceMatrix , PointClusterNumber , PointClusterNumberIndex ,Type  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                  
    Membership_value  =  set2List(Membership_value)
    Type  =  set2List(Type)
    count=0
    for  h  in  DistanceMatrix:
        count+=1
        #  if  count  ==  51  or  count  ==  101:
        #      print  ("/////////////////////////////////////////////////")
        if  count>0  and  count<51:
            c  =  0
            for  m  in  h:
                c+=1
                if  c>50  and  m<1.3:
                    print(m,count, c)

        

    #  print(DistanceMatrix  "\n"  DistanceMatrix.shape)
    print(Membership_value  "\n"  Type)
    return  PointClusterNumber,PointClusterNumberIndex-1,Membership_value,Type
  
  
  
def  ExpandClsuter(PointToExapnd    Membership_value    PointNeighbors    Cluster    MinumumPoints    Epsilon    Visited    DistanceMatrix  PointClusterNumber  PointClusterNumberIndex    Type    ):
    Neighbors=[]
    for  i  in  PointNeighbors:
        if  Visited[i]==0:
            Visited[i]=1
            Neighbors=numpy.where((DistanceMatrix[i]<Epsilon))[0]
            if  len(Neighbors)>=MinumumPoints:
                Type[i]  =  1
#                Neighbors  merge  with  PointNeighbors
                for  j  in  Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except  ValueError:
                        PointNeighbors.append(j)
                if  len(Neighbors)>=Max_value:
                    Membership_value[i]  =  1
                elif  len(Neighbors)<Max_value  and  len(Neighbors)>MinumumPoints:
                    Membership_value[i]  =  (len(Neighbors)  -  MinumumPoints)/(Max_value  -  MinumumPoints)
                else:
                    Membership_value[i]  =  0
            else:
                Type[i]  =  0
            if  PointClusterNumber[i]==0:
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
        
    return  Membership_value

def  calculate_PC(Data_len ,cluster_size,Membership_value):
    pc  =  0
    for  i  in  range(Data_len):
        pc  +=  Membership_value[i]**2
    return  pc/Data_len    

def  calculate_FPI(Data_len,cluster_size,Membership_value):
    value  =  0
    for  i  in  range(Data_len):
        value  +=  (Membership_value[i]**2)/Data_len
    FPI  =  1  -  (cluster_size/(cluster_size-1))*(1-value)
    return  FPI

#Generating  some  data  with  normal  distribution  at  
#(0  0)
#(8  8)
#(12  0)
#(15  15)
#  Data=GenerateData()

X  =  numpy.loadtxt("iris_data.txt")
Data  =  X[:  :-1]

#Adding  some  noise  with  uniform  distribution  
#X  between  [-3  17]  
#Y  between  [-3  17]
#  noise=scipy.rand(8  2)*20  -3
  
#  Noisy_Data=numpy.concatenate((Data  noise))
# size=20
  
  
# fig  =  plt.figure()
# ax1=fig.add_subplot(2  1  1)  #row    column    figure  number
# ax2  =  fig.add_subplot(212)
  
# ax1.scatter(Data[:  0]  Data[:  1]    alpha  =    0.5  )
#  ax1.scatter(noise[:  0]  noise[:  1]  color='red'    alpha  =    0.5)
#  ax2.scatter(noise[:  0]  noise[:  1]  color='red'    alpha  =    0.5)
  
  

Epsilon  =  1.2

MinumumPoints=8
Max_value  =  10
result  cluster_len  Membership_value  Type  =DBSCAN(Data  Epsilon  MinumumPoints  Max_value)
PC  =  calculate_PC(len(Data),cluster_len,Membership_value)
FPI  =  calculate_FPI(len(Data),cluster_len,Membership_value)
#printed  numbers  are  cluster  numbers
print  (result,PC,FPI)

#print  "Noisy_Data"
#print  Noisy_Data.shape
#print  Noisy_Data
  
#  for  i  in  range(len(result)):
#      ax2.scatter(Noisy_Data[i][0]  Noisy_Data[i][1]  color='yellow'    alpha  =    0.5)
        
#  plt.show()