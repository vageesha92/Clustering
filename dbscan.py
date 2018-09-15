from generate_data import (Point, get_sampling_func, points_list_to_array, array_to_points_list_after_scaling)
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as pl

def cluster_label(p):
    if p.predicted_cluster is None:
        return "Undefined"
    else:
        return str(p.predicted_cluster)


def set_cluster_label(p, cluster_name):
    p.predicted_cluster = str(cluster_name)


def check_if_noise(p, points, min_points, eps):
    # p should not be in any cluster
    # neighbours less than min_points
    if cluster_label(p) == "Undefined":
        if len(neighbours(p, eps, points)) < min_points:
            set_cluster_label(p, "Noise")


def neighbours(point, eps, points):
    # returns all points withing eps distance from p
    points_within_eps = []
    for other_point in points:
        distance = euclidean(point.toarray(), other_point.toarray())
        if distance <= eps:
            points_within_eps.append(other_point)
    return points_within_eps


def neighbours_excluding_point_itself(point, points):
    point_neighbours = neighbours(point, 0.1, points)
    neighbours_without_point = []
    for n in point_neighbours:
        if n != point:
            neighbours_without_point.append(n)
    return neighbours_without_point


def reachable_neighbours(points, min_points, eps, cluster_name, neighbours_excluding_point):
    for reachable_point in neighbours_excluding_point:
        if cluster_label(reachable_point) == "Noise":
            set_cluster_label(reachable_point, cluster_name)
        if cluster_label(reachable_point) != "Undefined":
            continue
        set_cluster_label(reachable_point, cluster_name)
        neighbours_reachable_point = neighbours(reachable_point, eps, points)
        if len(neighbours_reachable_point) >= min_points:
            neighbours_excluding_point.extend(neighbours_reachable_point)
    return neighbours_excluding_point


def find_all_points_in_cluster(points, cluster_name):
    points_in_cluster = []
    for point in points:
        if cluster_label(point) == str(cluster_name):
            points_in_cluster.append(point)
    return points_in_cluster


def dbscan(points, eps, min_points):
    cluster_name = 0
    clusters = {}
    for point in points:
        if cluster_label(point) != "Undefined":
            continue
        check_if_noise(point, points, min_points, eps)
        if cluster_label(point) == "Noise":
            continue
        cluster_name = cluster_name + 1
        set_cluster_label(point, cluster_name)
        cluster_neighbours = neighbours_excluding_point_itself(point, points)
        reachable_neighbours(points, min_points, eps, cluster_name, cluster_neighbours)
        clusters[cluster_name] = find_all_points_in_cluster(points, cluster_name)
    clusters["Noise"] = find_all_points_in_cluster(points, "Noise")
    return clusters


if __name__ == "__main__":
    points_list = [
        Point((0, 0, 0)),
        Point((0.002, 0, 0)),
        Point((0.1, 0, 0)),
        Point((10, 0, 0)),
        Point((11, 0, 0))
    ]
    get_samples = get_sampling_func(200, 2, 3)
    points_list, _ = get_samples()
    points_array = points_list_to_array(points_list)


    points_list = array_to_points_list_after_scaling(points_array)


    eps = 0.1
    minpts = 5
    dbsscan_output = dbscan(points_list, eps, minpts)
    pl.figure()
    colors = ['b', 'g', 'y']
    for cluster, cluster_points in dbsscan_output.items():
        cluster_points = points_list_to_array(cluster_points)
        if cluster != "Noise":
            pl.scatter(cluster_points[:,0], cluster_points[:, 1], c=colors.pop())
        else:
            pl.scatter(cluster_points[:, 0], cluster_points[:, 1], c="r")
    pl.show()