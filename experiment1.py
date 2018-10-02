from generate_data import random_points
from kmeans import k_means
from dbscan import dbscan


def dbscan_report(points, *args):
    eps = 0.1
    minpts = 5
    clusters_found = dbscan(points, eps, minpts)
    return len(clusters_found)


def kmeans_report(points, actual_num_clusters, *args):
    num_clusters_found = 0
    for i in range(5):
        centroids = k_means(actual_num_clusters, points)
        if len(centroids) > num_clusters_found:
            num_clusters_found = len(centroids)
    return num_clusters_found


def report(clustering_func_report):
    actual_num_clusters_list = [3, 5, 10, 15, 20]
    total_points_list = [500, 1000, 2000, 4000]
    dims = range(2, 102, 2)
    for total_points in total_points_list:
        for actual_num_clusters in actual_num_clusters_list:
            with open("{}_{}_{}.txt".format(clustering_func_report.__name__,
                                        total_points, actual_num_clusters), 'w') as report:
                for num_dimension in dims:
                    points = random_points(total_points, num_dimension, actual_num_clusters)
                    num_clusters = clustering_func_report(points, actual_num_clusters)
                    report.write("{:5}, \t{:5}\n".format(num_dimension, num_clusters))

report(dbscan_report)
