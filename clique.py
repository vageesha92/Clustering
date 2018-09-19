from itertools import product

import numpy as np
from generate_data import (Point, get_sampling_func,
                           array_to_points_list_after_scaling,
                           points_list_to_array)

class Unit:
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper
        self._points = []

    def is_point_in_unit(self, point):
        coord_in_range = [((self._lower[coord_index] is None) and True) or ((coord >= self._lower[coord_index]) and (coord < self._upper[coord_index]))
                          for coord_index, coord in enumerate(point.location)]
        return all(coord_in_range)

    def add_points_within_unit(self, points_list):
        for point in points_list:
            if self.is_point_in_unit(point):
                self._points.append(point)

    def is_dense(self, threshold):
        if len(self._points) >= threshold:
            return True

    def __str__(self):
        return "Unit: Lower = {}, Upper = {}".format(self._lower, self._upper)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

def partition(space, unit_length):
    # space : [[lower values], [upper values]]
    # space = [[0,0], [1,1]]
    # 0 <= unit_length < 1
    intervals_list = []
    for lower, upper in zip(*space):
        marks = np.arange(lower, upper, unit_length).tolist()
        if upper not in marks:
            marks.append(upper)
        interval = []
        for i, _ in enumerate(marks):
            if i < len(marks)-1:
                interval.append([marks[i], marks[i+1]])
        intervals_list.append(interval)
    units = []
    for rectangle in product(*intervals_list):
        lower = []
        upper = []
        for interval in rectangle:
            lower.append(interval[0])
            upper.append(interval[1])
        if all([l < u for l, u in zip(lower, upper)]):
            units.append(Unit(lower, upper))
    return units

def is_dense(unit):
    return unit.is_dense(5)

if __name__ == "__main__":
    import matplotlib.pyplot as pl
    units = partition([[0, 0], [1, 1]], 0.1)
    get_samples = get_sampling_func(200, 2, 3)
    points_list, _ = get_samples()
    points_array = points_list_to_array(points_list)
    points = array_to_points_list_after_scaling(points_array)
    for u in units:
        u.add_points_within_unit(points)
    dense_units = list(filter(is_dense, units))
    points_plotted = []
    for d in dense_units:
        print("{},\t len={}".format(d, len(d)))
        x = [d._lower[0], d._upper[0], d._upper[0], d._lower[0], d._lower[0]]
        y = [d._lower[1], d._lower[1], d._upper[1], d._upper[1], d._lower[1]]
        pl.plot(x, y, color='k')
        for point in d:
            point_array = point.toarray()
            pl.scatter(point_array[0], point_array[1])
            points_plotted.append(point)
    for point in points:
        if point not in points_plotted:
            point_array = point.toarray()
            pl.scatter(point_array[0], point_array[1], c="grey")
    pl.show()