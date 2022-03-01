class cluster:

    def __init__(self):
        pass

    def fit(self, X):
        pass

from itertools import count
import pandas as pd
import random
from statistics import mean

class KMeans(cluster):
    __k = 0
    __max_iters = 0

    # output
    labels_ = []
    cluster_centers_ = []

    def __init__(self, k = 5, max_iterations = 100):
        self.__k = k
        self.__max_iters = max_iterations
        print("k = %d, max_iter = %d \n" % (self.__k, self.__max_iters))
        return

    def fit(self, X):
        #df = pd.DataFrame(data = X, columns=['gender','x', 'y'])
        #print(type(df))
        #print(df)

        df = pd.read_csv('weight-height.csv')
        df = df.loc[:,['Height', 'Weight']]
        #data = [[0,0],[2,2],[0,2],[2,0],[10,10],[8,8],[10,8],[8,10]]
        #df = pd.DataFrame(data, columns=['Height','Weight'])
        x_max = df.iloc[:,[1]].max()
        x_min = df.iloc[:,[1]].min()
        y_max = df.iloc[:,[0]].max()
        y_min = df.iloc[:,[0]].min()
        print('x_min = %f, x_max = %f, y_min = %f, y_max = %f \n' % (x_min, x_max, y_min, y_max))

        #initialize centroids randomly
        centers = []
        centers_x = []
        centers_y = []
        centens_x_sum = [0] * self.__k          #sum of data points per cluster for x
        centens_y_sum = [0] * self.__k          #sum of data points per cluster for y
        cluster_data_count = [0] * self.__k     #how many data points per cluster
        count = 0
        while count < self.__k:
            center_x = random.choice(range(int(x_min), int(x_max)))
            centers_x.append(center_x)
            center_y = random.choice(range(int(y_min), int(y_max)))
            centers_y.append(center_y)
            #print("centers_x: ", centers_x)
            #print('centers_y: ', centers_y)
            count += 1
        centers.append(centers_x)
        centers.append(centers_y)
        print("centers: ", centers)
        #centers_df = pd.DataFrame(centers)
        #print(centers_df)

        sum_records = len(df)
        distances = []
        iter = 0
        while iter < self.__max_iters:
            print("iter: ", iter)
            count = 0

            #classify datas
            self.labels_.clear()         
            while count < sum_records:
                data_x = float(df.iloc[count,[1]])
                data_y = float(df.iloc[count,[0]])
                
                center_count = 0
                distances.clear()
                while center_count < self.__k:
                    distance_x = (data_x - centers_x[center_count]) ** 2
                    distance_y = (data_y - centers_y[center_count]) ** 2
                    distance = (distance_x + distance_y) ** 0.5
                    distances.append(distance)
                    center_count += 1
                #print("distances is:\n", distances)
                distances_min = min(distances)
                index = distances.index(distances_min)
                self.labels_.append(index)
                count += 1
            #print("one iteration, labels is:\n", self.labels_)
            
            #compute new centroids
            count = 0
            centens_x_sum = [0] * self.__k
            centens_y_sum = [0] * self.__k
            cluster_data_count = [0] * self.__k
            while count < sum_records:
                data_x = float(df.iloc[count,[1]])
                data_y = float(df.iloc[count,[0]])
                index_value = self.labels_[count]
                cluster_data_count[index_value] = cluster_data_count[index_value] + 1
                centens_x_sum[index_value] = centens_x_sum[index_value] + data_x
                centens_y_sum[index_value] = centens_y_sum[index_value] + data_y
                count += 1

            center_count = 0
            while center_count < self.__k:
                if cluster_data_count[center_count] == 0:   #if no data points in one cluster
                    centers_x[center_count] = 0
                    centers_y[center_count] = 0
                else:
                    centers_x[center_count] = centens_x_sum[center_count] / cluster_data_count[center_count]
                    centers_y[center_count] = centens_y_sum[center_count] / cluster_data_count[center_count]
                center_count += 1
            iter += 1
        
        print('centers_x is:\n', centers_x)
        print('centers_y is:\n', centers_y)
        #lastly, returns centroids
        #centroid = []
        center_count = 0
        while center_count < self.__k:
            #centroid.clear()
            centroid = []
            centroid.append(centers_x[center_count])
            centroid.append(centers_y[center_count])
            print("centroid: ", centroid)
            self.cluster_centers_.append(centroid)
            center_count += 1
        return

    def fitWithBalanced(self, X, balanced):
        print('[fit with banlanced]')
        df = pd.read_csv('weight-height.csv')
        df = df.loc[:,['Height', 'Weight']]
        x_max = df.iloc[:,[1]].max()
        x_min = df.iloc[:,[1]].min()
        y_max = df.iloc[:,[0]].max()
        y_min = df.iloc[:,[0]].min()
        print('x_min = %f, x_max = %f, y_min = %f, y_max = %f \n' % (x_min, x_max, y_min, y_max))

        #initialize centroids
        centers = []
        centers_x = []
        centers_y = []
        centens_x_sum = [0] * self.__k          #sum of data points per cluster for x
        centens_y_sum = [0] * self.__k          #sum of data points per cluster for y
        cluster_data_count = [0] * self.__k     #how many data points per cluster
        cluster_full = [0] * self.__k           #indicate whether the cluster is full
        count = 0
        while count < self.__k:
            center_x = random.choice(range(int(x_min), int(x_max)))
            centers_x.append(center_x)
            center_y = random.choice(range(int(y_min), int(y_max)))
            centers_y.append(center_y)
            #print("centers_x: ", centers_x)
            #print('centers_y: ', centers_y)
            count += 1
        centers.append(centers_x)
        centers.append(centers_y)
        print("centers: ", centers)
        #centers_df = pd.DataFrame(centers)
        #print(centers_df)

        sum_records = len(df)
        distances = []
        iter = 0
        while iter < self.__max_iters:
            print("iter: ", iter)
            count = 0

            #classify datas
            self.labels_.clear()
            cluster_data_count = [0] * self.__k
            cluster_full = [0] * self.__k         
            while count < sum_records:
                data_x = float(df.iloc[count,[1]])
                data_y = float(df.iloc[count,[0]])
                
                center_count = 0
                distances.clear()                
                while center_count < self.__k:
                    distance_x = (data_x - centers_x[center_count]) ** 2
                    distance_y = (data_y - centers_y[center_count]) ** 2
                    distance = (distance_x + distance_y) ** 0.5
                    distances.append(distance)
                    center_count += 1
                #check balanced
                distances_min = min(distances)
                distances_max = max(distances) 
                index = distances.index(distances_min)
                if balanced:
                    while cluster_full[index] == 1:
                        distances[index] = distances_max + 1
                        distances_min = min(distances)
                        index = distances.index(distances_min)
                    self.labels_.append(index)
                    cluster_data_count[index] = cluster_data_count[index] + 1
                    if cluster_data_count[index] >= (sum_records / self.__k):
                        cluster_full[index] = 1
                else:
                    self.labels_.append(index)
                count += 1
            print('cluster_data_count:\n', cluster_data_count)
            
            #compute new centroids
            count = 0
            centens_x_sum = [0] * self.__k
            centens_y_sum = [0] * self.__k
            cluster_data_count = [0] * self.__k
            while count < sum_records:
                data_x = float(df.iloc[count,[1]])
                data_y = float(df.iloc[count,[0]])
                index_value = self.labels_[count]
                cluster_data_count[index_value] = cluster_data_count[index_value] + 1
                centens_x_sum[index_value] = centens_x_sum[index_value] + data_x
                centens_y_sum[index_value] = centens_y_sum[index_value] + data_y
                count += 1

            center_count = 0
            while center_count < self.__k:
                if cluster_data_count[center_count] == 0:   #if no data points in one cluster
                    centers_x[center_count] = 0
                    centers_y[center_count] = 0
                else:
                    centers_x[center_count] = centens_x_sum[center_count] / cluster_data_count[center_count]
                    centers_y[center_count] = centens_y_sum[center_count] / cluster_data_count[center_count]
                center_count += 1
            iter += 1
        
        #lastly, returns centroids
        #centroid = []
        center_count = 0
        while center_count < self.__k:
            #centroid.clear()
            centroid = []
            centroid.append(centers_x[center_count])
            centroid.append(centers_y[center_count])
            self.cluster_centers_.append(centroid)
            center_count += 1
        return

    def fit_historical_weather(self, X):
        #df = pd.DataFrame(data = X, columns=['gender','x', 'y'])
        #print(type(df))
        #print(df)

        df = pd.read_csv('historical-weather.csv')
        df = df.loc[:,['relative_humidity', 'air_temp']]
        #data = [[0,0],[2,2],[0,2],[2,0],[10,10],[8,8],[10,8],[8,10]]
        #df = pd.DataFrame(data, columns=['Height','Weight'])
        x_max = df.iloc[:,[1]].max()
        x_min = df.iloc[:,[1]].min()
        y_max = df.iloc[:,[0]].max()
        y_min = df.iloc[:,[0]].min()
        print('x_min = %f, x_max = %f, y_min = %f, y_max = %f \n' % (x_min, x_max, y_min, y_max))

        #initialize centroids randomly
        centers = []
        centers_x = []
        centers_y = []
        centens_x_sum = [0] * self.__k          #sum of data points per cluster for x
        centens_y_sum = [0] * self.__k          #sum of data points per cluster for y
        cluster_data_count = [0] * self.__k     #how many data points per cluster
        count = 0
        while count < self.__k:
            center_x = random.choice(range(int(x_min), int(x_max)))
            centers_x.append(center_x)
            center_y = random.choice(range(int(y_min), int(y_max)))
            centers_y.append(center_y)
            #print("centers_x: ", centers_x)
            #print('centers_y: ', centers_y)
            count += 1
        centers.append(centers_x)
        centers.append(centers_y)
        print("centers: ", centers)
        #centers_df = pd.DataFrame(centers)
        #print(centers_df)

        sum_records = len(df)
        distances = []
        iter = 0
        while iter < self.__max_iters:
            print("iter: ", iter)
            count = 0

            #classify datas
            self.labels_.clear()         
            while count < sum_records:
                data_x = float(df.iloc[count,[1]])
                data_y = float(df.iloc[count,[0]])
                
                center_count = 0
                distances.clear()
                while center_count < self.__k:
                    distance_x = (data_x - centers_x[center_count]) ** 2
                    distance_y = (data_y - centers_y[center_count]) ** 2
                    distance = (distance_x + distance_y) ** 0.5
                    distances.append(distance)
                    center_count += 1
                #print("distances is:\n", distances)
                distances_min = min(distances)
                index = distances.index(distances_min)
                self.labels_.append(index)
                count += 1
            #print("one iteration, labels is:\n", self.labels_)
            
            #compute new centroids
            count = 0
            centens_x_sum = [0] * self.__k
            centens_y_sum = [0] * self.__k
            cluster_data_count = [0] * self.__k
            while count < sum_records:
                data_x = float(df.iloc[count,[1]])
                data_y = float(df.iloc[count,[0]])
                index_value = self.labels_[count]
                cluster_data_count[index_value] = cluster_data_count[index_value] + 1
                centens_x_sum[index_value] = centens_x_sum[index_value] + data_x
                centens_y_sum[index_value] = centens_y_sum[index_value] + data_y
                count += 1

            center_count = 0
            while center_count < self.__k:
                if cluster_data_count[center_count] == 0:   #if no data points in one cluster
                    centers_x[center_count] = 0
                    centers_y[center_count] = 0
                else:
                    centers_x[center_count] = centens_x_sum[center_count] / cluster_data_count[center_count]
                    centers_y[center_count] = centens_y_sum[center_count] / cluster_data_count[center_count]
                center_count += 1
            iter += 1
        
        print('centers_x is:\n', centers_x)
        print('centers_y is:\n', centers_y)
        #lastly, returns centroids
        #centroid = []
        center_count = 0
        while center_count < self.__k:
            #centroid.clear()
            centroid = []
            centroid.append(centers_x[center_count])
            centroid.append(centers_y[center_count])
            print("centroid: ", centroid)
            self.cluster_centers_.append(centroid)
            center_count += 1
        return

if __name__ == "__main__":
    kmeans = KMeans(k=4)
    #df = pd.read_csv('weight-height.csv')
    #print(df)
    #kmeans.fit(df)
    
    kmeans.fit(99)
    #kmeans.fitWithBalanced(99, True)
    #kmeans.fit_historical_weather(99)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)