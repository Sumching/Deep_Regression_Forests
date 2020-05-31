
from sklearn.cluster import KMeans
import operator
from functools import reduce
import numpy as np 

def kmeans(x, n_kluster=5):
    km = KMeans(n_clusters=n_kluster)
    km.fit(x)
    sigma = np.zeros([n_kluster, 1])

    for idx, label in enumerate(km.labels_):
        sigma[label] += (x[idx] - km.cluster_centers_[label]) ** 2

    for i in range(n_kluster):
        sigma[i] /= len(km.labels_[km.labels_==i])

    return np.reshape(km.cluster_centers_, [-1]), np.reshape(sigma, [-1])

if __name__ == "__main__":
    import numpy as np 
    
    x = np.random.random(10000).reshape(-1,1)
    mean, sigma = kmeans(x, 32)
    print(np.reshape(mean, [-1]), np.reshape(sigma, [-1]))