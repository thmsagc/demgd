import math
import warnings

class SparseGraph(object):
    def __init__(self, inputSet):
        self.edges = [[] for i in range(len(inputSet))]
        self.buckets = []
        self.diagonal = []

    def __binarySearch__(self, row, col):        
        start = 0
        mid = 0
        end = len(self.edges[row])-1
            
        if self.edges[row]:
            while start <= end:
                mid = (start + end) // 2
                if self.edges[row][mid][0] != col:
                    if start != end:
                        if self.edges[row][mid][0] > col:
                            end = mid
                        else:
                            start = mid+1             
                    else:
                        if self.edges[row][start][0] == col:
                            return [1, start]
                        
                        if self.edges[row][start][0] > col:
                            return [0, start]
                        else:
                            return [0, start+1]                
                else:
                    return [1, mid]
        return [0, 0]
    
    def __findDiagonal__(self):
        self.diagonal = []
        for i in range(len(self.edges)):
            for j in range(len(self.edges[i])):        
                if i == self.edges[i][j][0]:
                    self.diagonal.append(j)
                    
    def __insertEdge__(self, row, col, weight):
        if ((row or col) < len(self.edges)-1) or (row or col) > 0:        
            indice = self.__binarySearch__(row, col)
            if indice[0] == 0:
                self.edges[row].insert(indice[1], [col, weight])

                indice = self.__binarySearch__(col, row)
                if indice[0] == 0:
                    self.edges[col].insert(indice[1], [row, weight])
        else:
            warnings.warn("ERROR: Out of range. Limit: [0; " + str(len(self.edges)-1) + "]")
            
    def __removeEdge__(self, row, col):
        if ((row or col) < len(self.edges)-1) or (row or col) > 0:   
            if row == col:
                idEdge = self.__binarySearch__(row, col)
                if idEdge[0] == 1:
                    del self.edges[row][idEdge[1]]
                    return True
        
            idEdge = self.__binarySearch__(col, row)
            if idEdge[0] == 1:
                del self.edges[col][idEdge[1]]
                      
            idEdge = self.__binarySearch__(row, col)
            if idEdge[0] == 1:
                del self.edges[row][idEdge[1]]
                return True
            return False
        else:
            warnings.warn("ERROR: Out of range. Limit: [0; " + str(len(self.edges)-1) + "]")
        
    def __removeVertex__(self, index):
        if index < len(self.edges)-1 or index > 0:  
            length = len(self.edges[index])
            i = 0
            while i < length:
                if self.__removeEdge__(index, self.edges[index][i][0]):
                    length -= 1
                    continue
                else: i += 1
            del self.edges[index]
            self.__indexCorrector__(index)
        else:
            warnings.warn("ERROR: Out of range. Limit: [0; " + str(len(self.edges)-1) + "]")

    def __indexCorrector__(self, index):
        if index < len(self.edges)-1 or index > 0:  
            for i in range(len(self.edges)):
                for j in range(len(self.edges[i])):
                    if self.edges[i][j][0] > index:
                        self.edges[i][j][0] -= 1
        else:
            warnings.warn("ERROR: Out of range. Limit: [0; " + str(len(self.edges)-1) + "]")

class Deterministic_EMGD(object):
    def __init__(self, inputSet, distanceMetric):
        self.set = inputSet
        self.distanceMetric = distanceMetric
        self.marked = [False for i in self.set]
        self.removed = []

    def __applyKernelAndNormalize__(self):
        for i in range(len(self.graph.edges)):
            divisor = 0
            for j in range(len(self.graph.edges[i])):        
                k = self.__gaussianKernel__(self.graph.edges[i][j][1])
                self.graph.edges[i][j][1] = k
                divisor += k
                
            for j in range(len(self.graph.edges[i])):        
                self.graph.edges[i][j][1] /= divisor

                if i == self.graph.edges[i][j][0]:
                    self.graph.diagonal.append(j)

    def __bucketsVerify__(self, maxPerBucket):
        for i in range(len(self.buckets)):
            if len(self.buckets[i]) > maxPerBucket:
                return False
        return True

    def __buildBuckets__(self):        
        for i in range(len(self.set)):
            index = int(self.__pertinenceDegree__(self.set[i], self.min, self.max) * self.num_buckets)
            self.buckets[min(index, len(self.buckets)-1)].append(i)
        
    def __computeAverage__(self):
        trace = 0
        for i in range(len(self.graph.edges)):
            trace += self.graph.edges[i][self.graph.diagonal[i]][1] * len(self.graph.edges[i])
        return trace/len(self.graph.edges)

    def __computeMinMax__(self):
        dimension = len(self.set[0])
        size = len(self.set)
        self.min = []
        self.max = []

        for i in range(dimension):
            self.min.append(math.inf)
            self.max.append(-math.inf)

        for i in range(size):
            for j in range(dimension):
                if self.set[i][j] < self.min[j]:
                    self.min[j] = self.set[i][j]

                if self.set[i][j] > self.max[j]:
                    self.max[j] = self.set[i][j]

    def __computeStandardDeviation__(self):
        return math.sqrt(self.variancy)
    
    def __computeStatistics__(self):
        if self.set:
            self.average = self.__computeAverage__()
            self.variancy = self.__computeVariancy__()
            self.sdeviation = self.__computeStandardDeviation__()
        else:
            warnings.warn("ERROR: Empty set.")
            
    def __computeVariancy__(self):
        total = 0
        for i in range(len(self.graph.edges)):
            total += pow(self.graph.edges[i][self.graph.diagonal[i]][1] * len(self.graph.edges[i]) - self.average, 2)
        return total/len(self.graph.edges)        

    def __connectNeighbors__(self, index):
        if index < len(self.set)-1 or index > 0:  
            for i in range(1, len(self.graph.edges[index])-1):
                for j in range(i+1, len(self.graph.edges[index])):
                    if self.graph.edges[index][j][0] < len(self.set):
                        distance = self.__euclideanDistance__(self.set[self.graph.edges[index][i][0]], self.set[self.graph.edges[index][j][0]])
                        self.graph.__insertEdge__(self.graph.edges[index][i][0], self.graph.edges[index][j][0], distance)
        else: warnings.warn("ERROR: Out of range. Limit: [0; " + str(len(self.set)-1) + "]")

    def __createBuckets__(self, maxPerBucket):
        self.__computeMinMax__()
        self.num_buckets = ((len(self.set)-1) // maxPerBucket) * 4
        self.buckets = [[] for i in range(self.num_buckets)]
        self.__buildBuckets__()
        
        while not self.__bucketsVerify__(maxPerBucket):
            self.num_buckets *= 2
            self.buckets = [[] for i in range(self.num_buckets)]
            self.__buildBuckets__()
        self.__mergeBuckets__(maxPerBucket)

    def __createGraphKNN__(self, k, propagation=0):
        self.graph = SparseGraph(self.set)
        initial_propagation = max(0, -propagation)
         
        for i in range(len(self.buckets)):
            for p in range(max(0, i-propagation), min(len(self.buckets), i+propagation+1)):
                
                distances = [math.inf for i in range(k)]
                knn = [-1 for i in range(k)]
                
                for h in range(len(self.buckets[p])):   
                    for j in range(len(self.buckets[i])):
                                      
                        self.graph.__insertEdge__(self.buckets[i][j], self.buckets[i][j], 0)
                        dist = self.__euclideanDistance__(self.set[self.buckets[i][j]], self.set[self.buckets[p][h]])
                        
                        if dist < distances[k-1]:
                            distances[k-1] = dist
                            knn[k-1] = self.buckets[p][h]
                            pos = k-2
                            while pos >= 0 and dist <= distances[pos]:
                                distances[pos+1] = distances[pos]
                                knn[pos+1] = knn[pos]
                                distances[pos] = dist
                                knn[pos] = self.buckets[p][h]
                                pos-=1
                
                for f in range(k):
                    if knn[f] != -1:
                        self.graph.__insertEdge__(self.buckets[i][j], knn[f], distances[f])

        self.__applyKernelAndNormalize__()
        self.__computeStatistics__()

    def __deleteInstance__(self, index):
        if index < len(self.set)-1 or index > 0:  
            self.__connectNeighbors__(index)
            self.graph.__removeVertex__(index)
            del self.set[index]
            del self.graph.diagonal[index]
            del self.marked[index]
            self.graph.__findDiagonal__()
        else:
            warnings.warn("ERROR: Out of range. Limit: [0; " + str(len(self.edges)-1) + "]")

    def __distance__(self, p1, p2):
        if self.distanceMetric == "euclidean":
            return self.__euclideanDistance__(p1, p2)
        elif self.distanceMetric == "manhattan":
            return self.__manhattanDistanc__(p1,p2)
        else:
            return self.__euclideanDistance__(p1, p2)
        
    def __euclideanDistance__(self, p1, p2):
        distance = 0
        for i in range(len(p1)):
            distance += pow((p1[i]-p2[i]), 2)
        return distance ** 0.5
    
    def __EMGD__(self, multiplier):
        repeat = False
        if self.set:
            self.marked = [False for i in self.set]
            for i in range(len(self.graph.edges)):
                importance = self.graph.edges[i][self.graph.diagonal[i]][1] * len(self.graph.edges[i])
                if importance < self.average-(self.sdeviation*multiplier):
                    lessImportant = i
                    for j in range(len(self.graph.edges[i])):
                        neigh = self.graph.edges[i][j][0]
                        if self.marked[neigh]:
                            neighImportance = self.graph.edges[neigh][self.graph.diagonal[neigh]][1] * len(self.graph.edges[neigh])
                            if neighImportance < importance:
                                lessImportant = neigh
                            self.marked[neigh] = False
                    self.marked[lessImportant] = True
                    repeat = True
                    
            i = 0
            while(i < len(self.marked)):
                if self.marked[i]:
                    self.removed.append(self.set[i])
                    self.__deleteInstance__(i)
                    continue
                i += 1

        self.__computeStatistics__()       
        return repeat
              
    def __gaussianKernel__(self, value):
        return math.exp(-value*value)

    def __getRemoved__(self):
        return self.removed

    def __manhattanDistance__(self, p1, p2):
        distance = 0
        for i in range(len(p1)):
            distance += abs(p1[i]-p2[i])
        return distance
   
    def __mergeBuckets__(self, maxPerBucket):
        i = 0
        while i+1 < len(self.buckets):
            if len(self.buckets[i]) + len(self.buckets[i+1]) <= maxPerBucket:
                self.buckets[i] = self.buckets[i] + self.buckets[i+1]
                del self.buckets[i+1]
            else:
                i += 1
        
    def __pertinenceDegree__(self, xk, c1, c2):
        d_xk_c1 = self.__distance__(xk, c1)
        d_xk_c2 = self.__distance__(xk, c2)
        return d_xk_c1/(d_xk_c1 + d_xk_c2)
            
    def __iterator__(self, percentage, k):
        newlength = int(len(self.set) * percentage)
        while(newlength < len(self.set)):
            if not self.__EMGD__(k):
                break
            
def buildSetFromTxt(filename):
    with open(str(filename)+".txt", "r") as txt:
        return [line.strip() for line in txt]

def DeterministicEMGD(inputSet,percentage=0.75, maxPerBucket=25, k=5, propagation=1, multiplier=1, distanceMetric="euclidean"):
    if type(inputSet) is not list:
        warnings.warn("ERROR: The input set must be of the List type.")
        return
    if type(percentage) is not float:
        warnings.warn("ERROR: The param percentage must be of the float type.")
        return
    if type(k) is not int:
        warnings.warn("ERROR: The param k must be of the int type.")
        return
    if type(maxPerBucket) is not int:
        warnings.warn("ERROR: The param maxPerBucket must be of the int type.")
        return
    if type(propagation) is not int:
        warnings.warn("ERROR: The param propagation must be of the int type.")
        return
    if type(multiplier) is not int:
        warnings.warn("ERROR: The param multiplier must be of the int type.")
        return
    if type(distanceMetric) is not str:
        warnings.warn("ERROR: The param distanceMetric must be of the str type.")
        return
    
    if(len(inputSet)/2 < maxPerBucket):
        warnings.warn("ERROR: The maximum number of elements per bucket must be less than half the number of elements in the set.")
        maxPerBucket = int(len(inputSet)/2) // 2
        warnings.warn("Warning: Maximum value of elements per bucket changed implicitly to "+ str(maxPerBucket) +".")
    if(int(k) >= len(inputSet)):
        warnings.warn("ERROR: The maximum value for k must be at most the number of elements in the input set minus 1.")
        k = int(len(inputSet)-1)
        warnings.warn("Warning: The value of k changed implicitly to " + str(k) + ".")
    if(percentage > 1 or percentage < 0):
        warnings.warn("ERROR: The percentage parameter value must be within the range [0, 1].")
        percentage = float(0.75)
        warnings.warn("Warning: The value of percentage changed implicitly to " + str(percentage) + ".")
    if(multiplier < 1):
        warnings.warn("ERROR: The multiplier parameter must be greater than or equal to 1. .")
        multiplier = float(1)
        warnings.warn("Warning: The value of multiplier changed implicitly to " + str(multiplier) + ".")
        
    methodObject = Deterministic_EMGD(inputSet=list(inputSet), distanceMetric=str(distanceMetric))
    methodObject.__createBuckets__(int(maxPerBucket))
    methodObject.__createGraphKNN__(int(k),int(propagation))
    methodObject.__iterator__(float(percentage), int(multiplier))
    return methodObject.set, methodObject.__getRemoved__()
