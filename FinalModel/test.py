import pandas as pd
import numpy as np
import pickle
from math import radians,cos,sin,atan2,sqrt

# The dataset contains a list of trajectory points.THis functions return the percentage of abnormality in the dataset
# Marks point 1 if it is normal and 0 if it is abnormal

class GravityVector:
     def __init__(self,longitude,latitude,COG,SOG,medianDistance):
        self.longitude = longitude
        self.latitude = latitude
        self.COG = COG
        self.SOG = SOG
        self.medianDistance = medianDistance

class StoppingPoint:
    def __init__(self,lat,lng,COG,SOG):
        self.lat = lat
        self.lng = lng
        self.COG = COG
        self.SOG = SOG
class StoppingPointCluster:
    def __init__(self,clusterId,points):
        self.clusterId = clusterId
        self.points = points

def gpsDistance(lat1,lng1,lat2,lng2):
        earthRadius = 3958.75
        dLat = radians(lat2-lat1)
        dLng = radians(lng2-lng1)
        a = sin(dLat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLng / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = earthRadius * c
        meterConversion = 1609
        return abs(distance*meterConversion)

def ADD(row,SSPList):
    distance_arr = []
    for i in range(len(SSPList)):
        #for j in range(len(SSPList[i].points)):
        dist = gpsDistance(row.LATITUDE,row.LONGITUDE,SSPList[i].lat,SSPList[i].lng)
        distance_arr.append(dist)
    return distance_arr

def RDD(row,GVList):
    distance_arr = []
    for i in range(len(GVList)):
        for j in range(len(GVList[i])):
            dist = gpsDistance(row.LATITUDE,row.LONGITUDE,GVList[i][j].latitude,GVList[i][j].longitude)/abs(GVList[i][j].medianDistance)
            distance_arr.append(dist)
    return distance_arr

def CDD(row,GVList):
    distance_arr = []
    for i in range(len(GVList)):
        for j in range(len(GVList[i])):
            alpha = radians(row.COURSEOVERGROUND - GVList[i][j].COG)
            dist = cos(alpha) * (min(row.SPEEDOVERGROUND , GVList[i][j].SOG) / max(row.SPEEDOVERGROUND , GVList[i][j].SOG))
            distance_arr.append(dist)
    return distance_arr

def detectAbnormality(dataset , SSPList , GVList , add_threshold , rdd_threshold , cdd_threshold):
    dataset['label'] = 1
    count_abnormal_points = 0
    count_total_points = dataset.shape[0]
    for index ,row in dataset.iterrows():
        if(row.SPEEDOVERGROUND<=0.5):
            ADDL = ADD(row,SSPList)
            ADD_s = min(ADDL)
            #print("ADD_s " , ADD_s)
            if(ADD_s>add_threshold):
                dataset.loc[index,'label'] = 0
                count_abnormal_points += 1
        else:
            RDDL = RDD(row,GVList)
            RDD_m = min(RDDL)
            #print("RDD_m" ,RDD_m)
            if(RDD_m>rdd_threshold):
                dataset.loc[index,'label'] = 0
                count_abnormal_points += 1
            else:
                CDDL = CDD(row,GVList)
                CDD_m = max(CDDL)
                #print("CDD_m" , CDD_m)
                if(CDD_m<cdd_threshold):
                    dataset.loc[index,'label'] = 0
                    count_abnormal_points += 1
    abnormality = count_abnormal_points/count_total_points
    return abnormality
    
abnormality_score = []
test_data = pd.read_csv('dataa.csv')
fileName = 'GravityVector'
fileObject = open(fileName,'rb') 
GV = pickle.load(fileObject) 
fileObject.close()
fileName = 'SSPVector'
fileObject = open(fileName,'rb') 
SSP = pickle.load(fileObject)
fileObject.close()
arr = test_data.MMSI.unique()
for i in range(len(arr)): 
    test_data1 = test_data[test_data.MMSI==arr[i]]
    x = detectAbnormality(test_data1,SSP,GV,1000000,10000,0.95)
    abnormality_score.append(x)
    print("Abnormality score of " , end=" ")
    print(arr[i],end="")
    print("th  trajectory is",end="  ")
    print(x)