# generation of synthetic data

import numpy as np
from sklearn.preprocessing import minmax_scale

class Clusters_Collection:
    def __init__(self, clusters=None):
        if clusters is None:
            self.clusters = []
        else:
            self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def all_points(self):
        points_list = []
        for cluster in self.clusters:
            for point in cluster:
                points_list.append(point)
        return points_list

    def append(self, new_cluster):
        self.clusters.append(new_cluster)

class Point:

    def __init__(self, location):
        self.location = location
        self.predicted_cluster = None

    def __eq__(self, other):
        if self.location == other.location:
            return True

    def __str__(self):
        return "Point: location = {}".format(",".join(str(l) for l in self.location))

    def toarray(self):
        return np.array(self.location)

    def dimension(self):
        return len(self.location)

    @classmethod
    def random_around_center(cls, center, spread):
        dim = center.dimension()
        location = (spread * np.random.randn(1, dim)) + center.toarray()
        return Point(location)


def points_list_to_array(points_list):
    array_list = []
    for point in points_list:
        array_list.append(point.toarray())
    return np.vstack(array_list)

def array_to_points_list_after_scaling(array_of_points):
    array_of_points = minmax_scale(array_of_points)
    points_list = []
    for point_as_array in array_of_points:
        points_list.append(Point(tuple(point_as_array.tolist())))
    return points_list

def get_sampling_func(num_total_points=10, dim=2, num_clusters=2, noise=None):
    def sampling_func():
        def get_centers(range_to_split, num_centers):
            first_dim = np.linspace(range_to_split[0], range_to_split[1], num_centers).tolist()
            all_dims = [first_dim]
            for _d in range(1, dim):
                dim_values = np.random.randint(range_to_split[0], range_to_split[1], num_centers).tolist()
                all_dims.append(dim_values)
            #xs = np.linspace(range_to_split[0], range_to_split[1], num_centers).tolist()
            #ys = np.random.randint(range_to_split[0], range_to_split[1], num_centers).tolist()
            ps = [p for p in zip(*all_dims)]
            centers = [Point(center) for center in ps]
            return centers

        def random_variance():
            return np.random.randint(1, 5)


        def gaussian_noise(data):
            if not noise:
                return data
            noise_data = noise * np.random.randn(len(data))
            data = data + noise_data
            return data

        cluster_centers = get_centers([-50, 50], num_clusters)
        all_clusters = Clusters_Collection()
        for cluster_index in range(num_clusters):
            # create random points around this cluster
            # TODO: change number of points in each cluster . Don't use uniform number of points in all clusters
            cluster = [Point.random_around_center(cluster_centers[cluster_index], random_variance()) for num_points in range(num_total_points//num_clusters)]
            # append this cluster to the end of all cluster list
            all_clusters.append(cluster)

        # stack vertically the items in mixture distribution into a single array : converting list to array
        return all_clusters.all_points(), cluster_centers

    return sampling_func


if __name__ == "__main__":
    import matplotlib.pylab as pl
    samples = get_sampling_func(1000, 2, 5)
    data, cluster_centers = samples()
    data = points_list_to_array(data)

    pl.figure()
    pl.scatter(data[:, 0], data[:, 1])
    pl.show()
