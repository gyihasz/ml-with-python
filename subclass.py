import time
import inspect
import numpy as np 

class NearestSubClassCentroidClassifier:
    def train(data, labels, properties = None):
        CR = NearestSubClassCentroidClassifier
        #Find X clusters for each category points and then flatten the model list
        model = [{'class': category, 'coor': cluster } for category in set(labels) for cluster in CR._get_clusters_for_category_points(data, labels, category, properties)]
        return model


    def use(model, data_point, properties = None):
        #Calculate distance to each category
        distances = [np.linalg.norm(category['coor'] - data_point.flatten()) for category in model]
        #Find index of minimum distance
        model_index = distances.index(min(distances))
        #Get category corrosponding to minimum distance
        category = model[model_index]['class']
        #Return closest category
        return category


    def _get_clusters_for_category_points(data, labels, category, properties):
        #Find points in category
        category_points = NearestSubClassCentroidClassifier._get_points_from_category(data, labels, category)
        #Flatten points
        flattened_points = NearestSubClassCentroidClassifier._flatten_points(category_points) 
        #Find clusters
        clusters = NearestSubClassCentroidClassifier._k_mean_clustering(flattened_points, properties['nr_clusters'])
        return clusters


    def _get_points_from_category(points, labels, category):
        #Find points in category
        return [points[index] for index in NearestSubClassCentroidClassifier._indexes_w_cat(labels, category)]
    

    def _indexes_w_cat(labels, category):
        #Find labels in category
        return [index for index, value in enumerate(labels) if value == category]


    def _flatten_points(points):
        #Flatten points
        return [point.flatten() for point in points]


    def _k_mean_clustering(data, nr_clusters, iterations = 10):
        return KMeanClustering.find_clusters(data, nr_clusters, iterations)


class KMeanClustering:
    def find_clusters(data, nr_clusters, iterations = 10):
        clusters = KMeanClustering._generate_random_centers(data, nr_clusters)
        for _ in range(iterations):
            #Attach point to the closest cluster  
            clusters_points = KMeanClustering._attach_points_to_clusters(data, clusters)
            #Calculate mean based on points 
            clusters = KMeanClustering._calculate_mean_points(clusters_points)
        return clusters

    
    def _generate_random_centers(data, nr_clusters):
        nr_of_dimensions = data[0].shape
        return [np.random.random_sample(size=(nr_of_dimensions))*np.max(data) for _ in range(nr_clusters)]


    def _attach_points_to_clusters(data, clusters):
        #Get list of indexes for the closest cluster to each point
        closest_cluster_list = [KMeanClustering._get_closest_cluster_index(point, clusters) for point in data]
        #Split points into clusters
        clusters_points = [[] for _ in clusters]
        for index, point in enumerate(data):
            cluster_index = closest_cluster_list[index]
            clusters_points[cluster_index].append(point)
        return clusters_points
    
    
    def _get_closest_cluster_index(point, clusters):
        #Find distance for each cluster
        distance_to_clusters = [KMeanClustering._distance(cluster, point) for cluster in clusters]
        #find closest cluster index
        closest_cluster_index = distance_to_clusters.index(min(distance_to_clusters))
        return closest_cluster_index

    def _distance(point_a, point_b):
        return np.linalg.norm(point_a - point_b)  


    def _calculate_mean_points(clusters_points):
        return [np.mean(cluster, axis=0) for cluster in clusters_points]

class MachineLearning:
    def train(ml_class, data, labels, properties = None):
        #Start time for stat measurement
        start_time = time.time()
        #Train model
        model = ml_class.train(data, labels, properties)
        #Calculate stats
        train_stats = {'time': time.time() - start_time}
        #Return model and stats
        return model, train_stats

    def use(ml_class, model, data_point, properties = None):
        #Use the trained model to find a label
        point_class = ml_class.use(model, data_point, properties)
        return point_class

    def get_accuracy(ml_class, model, data_points, data_labels, properties = None):
        #Check if user have split model and stats
        if isinstance(model, tuple) and len(model) == 2 and isinstance(model[1], type({})):
            model = model[0]
        #Start timer used to claculate stats
        #start_time = time.time()
        #Call 'use' for each datapoint and match it with a label
        prediction_list = [MachineLearning.use(ml_class, model, point, properties) for point, label in zip(data_points, data_labels)]
        return prediction_list

    
class NearestClassifiers:
    class SubClassCentroid:
        def train(data_train, labels, properties = 1):
            return MachineLearning.train(NearestSubClassCentroidClassifier, data_train, labels, properties)

        def use(model, data_point):
            return MachineLearning.use(NearestSubClassCentroidClassifier, model, data_point)

        def test(model, data_test, labels_test):
            return MachineLearning.get_accuracy(NearestSubClassCentroidClassifier, model, data_test, labels_test)