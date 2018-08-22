# generation of synthetic data

import numpy as np



def get_sampling_func(num=10, dim=2, mixture=2):
    def sampling_func():
        def partition(range_to_split, num_partitions):
            xs = np.linspace(range_to_split[0], range_to_split[1], num_partitions).tolist()
            ys = np.random.randint(range_to_split[0], range_to_split[1], num_partitions).tolist()
            ps = np.array([[x, y] for x, y in zip(xs, ys)])
            return ps
        cluster_centers = partition([-20, 20], mixture)
        mixture_distributions = []
        for m in range(mixture):
            # create a distribution with random mean
            single_distribution = np.random.randn(num // mixture, dim) + cluster_centers[m]
            # append it to the end of mixture distribution list
            mixture_distributions.append(single_distribution)
        # stack vertically the items in mixture distribution into a single array : converting list to array
        return np.vstack(mixture_distributions), cluster_centers
    return sampling_func


if __name__ == "__main__":
    import matplotlib.pylab as pl
    sample_1000 = get_sampling_func(1000, 2, 3)
    data, cluster_centers = sample_1000()
    pl.figure()
    pl.scatter(data[:, 0], data[:, 1])
    pl.show()
