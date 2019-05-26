import  numpy  as  numpy
import  scipy  as  scipy
from  sklearn  import  cluster
import  matplotlib.pyplot  as  plt
import time
  
     
def  set2List(NumpyArray):
    list  =  []
    for  item  in  NumpyArray:
        list.append(item.tolist())
    return  list
  

def  member(value,MinumumPoints,Max_Points):
    if  value<=MinumumPoints:
        return  0
    elif  value<Max_Points:
        return  (value  -  MinumumPoints)/(Max_Points  -  MinumumPoints)
    else:
        return  1    

def  DBSCAN(Dataset,Epsilon_min,Epsilon_max, MinumumPoints,Max_Points,DistanceMethod  =  'euclidean'):
#    Dataset  is  a  mxn  matrix    m  is  number  of  item  and  n  is  the  dimension  of  data
    m,n = Dataset.shape
    Visited = numpy.zeros(m,'int')
    Type = numpy.zeros(m)
    Membership = []
    for i in range(m):
        Membership.append({})
#  
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=numpy.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix  =  scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset,DistanceMethod))
   
    for  i  in  range(m):
        if  Visited[i] == 0:
            Visited[i] = 1
            PointNeighbors = numpy.where(DistanceMatrix[i]<Epsilon_max)[0]
            PointNeighbors = set2List(PointNeighbors)
            

            density = 0
            for  k  in  PointNeighbors:
                membership  =  0
                if DistanceMatrix[i][k] <= Epsilon_min:
                    membership  =  1

                elif DistanceMatrix[i][k] <= Epsilon_max:
                    membership = ((Epsilon_max - DistanceMatrix[i][k])/(Epsilon_max - Epsilon_min))
                
                else:
                    membership  =  0

                density = density + membership      
            # print(density)
            if  member(density,MinumumPoints,Max_Points)  ==  0  :
                Type[i] = -1
                Visited[i] = 0

            else:
                for k in PointNeighbors:
                    if PointClusterNumber[k]!=PointClusterNumberIndex and Type[k] == 0 and k!=i:
                        Visited[k] = 0
                Cluster = []
                Cluster.append(i)
                PointClusterNumber[i] = PointClusterNumberIndex
                Type[i] = 1
                Membership[i][PointClusterNumberIndex] = round(member(density ,MinumumPoints ,Max_Points),2)
                
                ExpandClsuter(i,PointNeighbors,Cluster,MinumumPoints,Max_Points,Epsilon_min,Epsilon_max,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex,Type,Membership  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                  
    # print(Type,"\n",Membership)
    return  PointClusterNumber,PointClusterNumberIndex-1,Membership,Type  
  
  
  
def  ExpandClsuter(PointToExapnd,PointNeighbors,Cluster,MinumumPoints,Max_Points,Epsilon_min,Epsilon_max,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex,Type,Membership):
    Neighbors=[]

    for  i  in  PointNeighbors:
        if  Visited[i]==0:
            Visited[i]=1
            Neighbors=numpy.where(DistanceMatrix[i]<Epsilon_max)[0]
            # print("Expand::",i)
            # print(Neighbors)
            
            density  =  0
            for  k  in  PointNeighbors:
                membership  =  0
                if  DistanceMatrix[i][k]<=Epsilon_min:
                    membership  =  1

                elif  DistanceMatrix[i][k]<=Epsilon_max:
                    membership  =  ((Epsilon_max-DistanceMatrix[i][k])/(Epsilon_max-Epsilon_min))
                
                else:
                    membership  =  0
                density  =  density  +  membership
            m  =  round(member(density,MinumumPoints,Max_Points),2)
            # print(density)      
            if  m  >  0:
                for k in Neighbors:
                    if PointClusterNumber[k]!=PointClusterNumberIndex and Type[k] == 0 and k!=i:
                        Visited[k] = 0
                for  j  in  Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except  ValueError:
                        PointNeighbors.append(j)
                Type[i]  =  1
                Membership[i][PointClusterNumberIndex]  =  m
            else:
                Type[i]  =  0
                minimum  =  9999
                for  k  in  Neighbors:
                    if  k  !=  i:
                        if  DistanceMatrix[i][k]  <  Epsilon_min:
                            h  =  1
                        else  :
                            h  =  (Epsilon_max  -  DistanceMatrix[i][k])  /  (Epsilon_max  -  Epsilon_min)
                        Neighbors2=numpy.where(DistanceMatrix[k]<Epsilon_max)[0]
                        density  =  0
                        for  p  in  Neighbors2:
                            membership  =  0
                            if  DistanceMatrix[k][p]<=Epsilon_min:
                                membership  =  1

                            elif  DistanceMatrix[p][k]<=Epsilon_max:
                                membership  =  ((Epsilon_max-DistanceMatrix[i][k])/(Epsilon_max-Epsilon_min))
                            
                            else:
                                membership  =  0
                            density  =  density  +  membership
                        m  =  round(member(density,MinumumPoints,Max_Points),2)
                        mini  =  9999
                        if  m>0  and  h>0:
                            mini  =  min(m, round(h,2))
                        if  mini  <  minimum:
                            minimum  =  mini
                if minimum == 9999:
                    Membership[i][PointClusterNumberIndex]  =  0
                else:
                    Membership[i][PointClusterNumberIndex] = minimum                                    


        if  PointClusterNumber[i]==0:
            Cluster.append(i)
            PointClusterNumber[i]=PointClusterNumberIndex
    return
  

def  calculate_PC(Data_len,cluster_size,Membership_value):
    pc  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                pc  +=  Membership_value[i][j+1]**2
    return  pc/Data_len  

def  calculate_FPI(Data_len,cluster_size,Membership_value):
    value  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                value  +=  (Membership_value[i][j+1]**2)/Data_len
    if cluster_size == 1:
        return 0
    else:
        FPI  =  1  -  (cluster_size/(cluster_size-1))*(1-value)
        return  FPI          

def  F_recall(Output,Membership_final,k):
    membership_sum = 0
    c = 0
    for i in  range(len(Membership_final)):
        if k in Membership_final[i] and Output[i] == k:
            membership_sum += Membership_final[i][k]
    for i in Output:
        if k == i:
            c += 1
    if membership_sum == 0:
        return 0,c    
    return membership_sum/c,c            

def  F_Precision(Membership_final,k):
    membership_sum = 0
    c = 0
    for i in  range(len(Membership_final)):
        if k in Membership_final[i] and Output[i] == k:
            membership_sum += Membership_final[i][k]
    for i in Membership_final:
        if k in i:
            c += 1
    if c == 0:
        return 0,c        
    return membership_sum/c,c

def Combine_clusters(Membership_value,size,Points):
    cluster_size = size
    m = 1
    l = 1
    for j in range(1,size+1):
        for k in range(1,j):
            count = 0
            flag = 0
            for i in Membership_value:
                if (j in i) and (k in i):
                    count += 1
                if count == Points:
                    flag = 1
                    break
            if flag == 1:
                cluster_size -= 1
                l = min(j,k)
                for i in Membership_value:
                    if (j in i) and (k in i):
                        i[l] = max(i[j],i[k])
                        if l!=k:    
                            del i[k]
                        if l!=j:    
                            del i[j]
                    elif k in i:
                        i[l] = i[k]
                        if l!=k:    
                            del i[k]
                    elif j in i:
                        i[l] = i[j]
                        if l!=j:    
                            del i[j]                  

    return Membership_value,cluster_size                            
def membership_set(Membership,r1):
    set_value = []
    for i in range(len(Membership)):
        for j in range(r1+1):
            if j in Membership[i]:
                set_value.append(j)
    return set_value            

def mapping_cluster(Membership,set_result,Output,set_output):
    count = {}
    c =0
    Result = []
    for i in range(len(Membership)):
        Result.append(0)
    for i in set_result:
        # for m in set_output:
        #         count[m] = 0
        max1 = 0
        pos = 0     
        for j in set_output:
            c = 0
            for k in range(len(Membership)):
                if i in Membership[k] and j == Output[k]:
                    c += 1
            if c>max1:
                max1 = c
                pos = j
        for j in range(len(Membership)):
            if i in Membership[j]:
                Result[j] = pos
                Membership[j][pos] = Membership[j][i]
                if pos != i:
                    del Membership[j][i]
    return Membership,Result      

def accuracy(Membership,Output):
    count = 0
    for i in range(len(Output)):
        if Output[i] in Membership[i]:
            count+=1
    return count/len(Output)
def Plot_cluster(Membership,Data,Output,file_name):
    m = max(Output)
    color = {1:'red',2:'blue',3:'yellow',4:'brown',5:'green',6:'orange',7:'cyan',8:'olive',9:'gray',10:'pink',11:'lime',12:'blueviolet',13:'gold',14:'aqua',15:'maroon',16:'wheat',17:'peru',18:'darkkhaki',19:'red',20:'blue',21:'yellow',22:'brown',23:'green',24:'orange',25:'cyan',26:'olive',27:'gray',28:'pink',29:'lime',30:'blueviolet',31:'gold'} 
    plt.style.use('ggplot')
    for i in range(len(Output)):
        if Output[i] in Membership[i]:
            plt.scatter(Data[i][0],Data[i][1],color = color[Output[i]])
        else:
            plt.scatter(Data[i][0],Data[i][1],color ='black')
    plt.title(file_name)
    plt.show()
file_name = "compound"
X  =  numpy.loadtxt(file_name+".txt",dtype  =  'float')
Data  =  X[:,  :-1]
Output  =  X[: ,-1]

count = 0
Epsilon_min= 1.9
Epsilon_max = 2.15
MinumumPoints = 5
Max_Points = 5
t0 = time.time()
result,cluster_len,Membership,Type=DBSCAN(Data,Epsilon_min,Epsilon_max,MinumumPoints,Max_Points)
t1 = time.time()
set_value = membership_set(Membership,cluster_len)  
Membership_final,Result = mapping_cluster(Membership,set_value,Output,list(set(Output)))        

# for i in range(len(Membership_final)):
#     print(Membership_final[i],Output[i],"\n")

PC = calculate_PC(len(Data),cluster_len,Membership_final)
FPI = calculate_FPI(len(Data),cluster_len,Membership_final)

F_measure  =  0
l  =  list(set(Output))
for i in l:
    try:
        f_Precision,Cj = F_Precision(Membership_final,i)
        f_recall,Dc = F_recall(Output,Membership_final,i)
        f_measure = 2*(f_Precision*f_recall)/(f_recall+f_Precision)
        F_measure += (Cj/len(Data))*f_measure
    except:
        F_measure +=0
Accuracy = accuracy(Membership,Output)
print("File_name:  ",file_name,"\n","Accuracy: " ,Accuracy,"\n","PC:  ",PC,"\n","FPI:  ",FPI,"\n","F_measure:  ",F_measure,"\n","Time_complexity:  ",t1-t0)
Plot_cluster(Membership,Data,Output,file_name)

# q1,q2,q3,q4 = 0,0,0,0
# max_F_measure,max_Pc,max_FPI = 0,0,0
# Epsilon_min = .9
# for z in range(2):
#     Epsilon_max=Epsilon_min+.05
#     for x in range(2):
#         MinumumPoints=8
#         for q in range(2):
#             Max_Points=MinumumPoints 
#             for p in range(2):
#                 result,cluster_len,Membership,Type=DBSCAN(Data,Epsilon_min,Epsilon_max,MinumumPoints,Max_Points)
#                 set_value = membership_set(Membership,cluster_len)  
#                 Membership_final,Result = mapping_cluster(Membership,set_value,Output,list(set(Output)))        

#                 # for i in range(len(Membership_final)):
#                 #     print(Membership_final[i],Output[i],"\n")

#                 PC = calculate_PC(len(Data),cluster_len,Membership_final)
#                 FPI = calculate_FPI(len(Data),cluster_len,Membership_final)

#                 F_measure  =  0
#                 l  =  list(set(Output))
#                 for i in l:
#                     try:
#                         f_Precision,Cj = F_Precision(Membership_final,i)
#                         f_recall,Dc = F_recall(Output,Membership_final,i)
#                         f_measure = 2*(f_Precision*f_recall)/(f_recall+f_Precision)
#                         F_measure += (Cj/len(Data))*f_measure
#                     except:
#                         F_measure +=0
#                 # print(PC,FPI,F_measure)
#                 if F_measure>max_F_measure:
#                     max_F_measure = F_measure 
#                     q1 = Epsilon_min
#                     q2 = Epsilon_max
#                     q3 = MinumumPoints
#                     q4 = Max_Points  
#                     if PC>max_Pc:
#                         max_Pc = PC
#                         if FPI>max_FPI:
#                             max_FPI = FPI
#                 count+=1  
#                 # print(count)   
#                 Max_Points += 1
#             MinumumPoints += 1 
#         Epsilon_max += .1
#     Epsilon_min += .1   

# print("\n\n\n\n")         
# print(max_Pc,max_FPI,max_F_measure,q1,q2,q3,q4)