from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
data_file = open('data/data/iris.data', 'r')
data = []
first_line = True
for line in data_file:
    if not first_line:
        sample = line.split(',')
        new_sample = []
        for i in range(4):
            new_sample.append(float(sample[i]))
        data.append(new_sample)
    else:
        first_line = False
data = np.array(data)

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

def k_means(C):
    
    # Write your code here!
    error = 1
    centroid_assignments = np.zeros(len(data))
    C_final = np.copy(C)

    while error > 0.001:
        # Assign data points to centroids
        for i in range(len(data)):
            
            # Initialize closest centroid
            closest_centroid = 0
            shortest_distance = data[i]-C[0]
            shortest_distance = np.sum(shortest_distance*shortest_distance)
            
            # Iterate to find the closest centroid
            for j in range(1, k, 1):
                current_distance = data[i]-C[j]
                current_distance = np.sum(current_distance*current_distance)
                if(current_distance < shortest_distance):
                    shortest_distance = current_distance
                    closest_centroid = j
                    
            centroid_assignments[i] = closest_centroid
        
        # Move centroids to average position
        for i in range(k):
            points = 0
            new_center = np.zeros(4)
            for j in range(len(data)):
                if(centroid_assignments[j] == i):
                    points += 1
                    new_center += data[j]
            C_final[i] = new_center * (1/points)
            
        # Calculate error
        error = np.sum(np.absolute(C_final-C))
        C = np.copy(C_final)

    return C_final

C = k_means(C)
print("Final Centers")
print(C)




