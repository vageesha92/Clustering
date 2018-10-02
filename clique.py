from itertools import (product, combinations, chain, tee)
from multiprocessing import Pool
from functools import lru_cache, reduce
from operator import add, attrgetter
import numpy as np
import matplotlib.pyplot as pl
from generate_data import random_points, Point


class Unit:
    def __init__(self, lower, upper, threshold):
        self._lower = lower
        self._upper = upper
        self._points = []
        self.cluster_number = "Undefined"
        self._threshold = threshold

    def is_point_in_unit(self, point):
        coord_in_range = [((self._lower[coord_index] is None) and True) or
                          ((coord >= self._lower[coord_index]) and (coord < self._upper[coord_index]))
                          for coord_index, coord in enumerate(point.location)]
        return all(coord_in_range)

    def subspace(self):
        subspace_numbers = []
        for subspace_number, lower_range in enumerate(self._lower, 1):
            if lower_range is not None:
                subspace_numbers.append(subspace_number)
        return subspace_numbers

    def add_points_within_unit(self, points_list):
        for point in points_list:
            if self.is_point_in_unit(point):
                points_list.remove(point)
                self._points.append(point)
        return self

    def is_dense(self):
        if len(self) >= self._threshold:
            return True

    def __str__(self):
        return "Unit: Lower = {}, Upper = {}, subspace={}, len={}".format(self._lower,
                                                             self._upper, self.subspace(),
                                                             len(self))

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._points)

    def __eq__(self, other):
        return (self._lower == other._lower) and (self._upper == other._upper)

    def __bool__(self):
        return True

    def plot(self):
        x = [self._lower[0], self._upper[0], self._upper[0], self._lower[0], self._lower[0]]
        y = [self._lower[1], self._lower[1], self._upper[1], self._upper[1], self._lower[1]]
        pl.plot(x, y, color='k')


def partition(space, unit_length, threshold):
    # space : [[lower values], [upper values]]
    # space = [[0,0], [1,1]]
    # 0 <= unit_length < 1
    intervals_list = []
    for lower, upper in zip(*space):
        if (lower is None) and (upper is None):
            intervals_list.append([None])
            continue
        marks = np.round(np.arange(lower, upper, unit_length), 1).tolist()
        if upper not in marks:
            marks.append(upper)
        interval = []
        for i, _ in enumerate(marks):
            if i < len(marks)-1:
                interval.append([marks[i], marks[i+1]])
        intervals_list.append(interval)
    #units = []
    for rectangle in product(*intervals_list):
        lower = []
        upper = []
        for interval in rectangle:
            if interval is None:
                lower.append(None)
                upper.append(None)
            else:
                lower.append(interval[0])
                upper.append(interval[1])
        if all([(((l is None) and (u is None) and True)
                 or (l < u)) for l, u in zip(lower, upper)]):
            #units.append(Unit(lower, upper))
            yield Unit(lower, upper, threshold)
    #return units


def plot_dense_units(dense_units):
    for d in dense_units:
        print("{},\t len={}".format(d, len(d)))
        d.plot()
        for point in d:
            point.plot(c="blue")


def plot_not_dense_points(points, dense_units):
    for p in points:
        if p not in dense_units:
            p.plot(c="grey")


def get_units(subspace, num_dimensions, unit_length, threshold):
    lower = []
    upper = []
    for i in range(1, num_dimensions+1):
        if i in subspace:
            lower.append(0)
            upper.append(1)
        else:
            lower.append(None)
            upper.append(None)
    return partition([lower, upper], unit_length, threshold)


def get_subspaces(num_dimensions):
    dimensions = range(1, num_dimensions+1)
    subspaces = []
    for i in dimensions:
        subspaces.append(list(combinations(dimensions, i)))
    return subspaces


def can_join_units(unit1, unit2):
    unit1_subspace = unit1.subspace()
    unit2_subspace = unit2.subspace()
    if len(unit1_subspace) != len(unit2_subspace):
        return False
    for dimension1, dimension2 in zip(unit1_subspace[:-1], unit2_subspace[:-1]):
        if ((dimension1 != dimension2) or
                (unit1._lower[dimension1-1] != unit2._lower[dimension2-1]) or
                (unit1._upper[dimension1-1] != unit2._upper[dimension2-1])):
            return False
    if unit1_subspace[-1] == unit2_subspace[-1]:
        return False
    return True


def join(unit1, unit2):
    lower = list(unit1._lower)
    upper = list(unit1._upper)
    unit2_subspace = unit2.subspace()
    lower[unit2_subspace[-1]-1] = unit2._lower[unit2_subspace[-1]-1]
    upper[unit2_subspace[-1]-1] = unit2._upper[unit2_subspace[-1]-1]
    return Unit(lower, upper, unit1._threshold)

def projection(unit):
    unit_subspace = unit.subspace()
    dimension_minus_one = len(unit_subspace) - 1
    subspaces_minus_one_dimension = combinations(unit_subspace, dimension_minus_one)
    projection_units = []
    for subspace in subspaces_minus_one_dimension:
        lower = [None for i in range(len(unit._lower))]
        upper = [None for i in range(len(unit._upper))]
        for i in subspace:
            lower[i-1] = unit._lower[i-1]
            upper[i-1] = unit._upper[i-1]
        projection_units.append(Unit(lower, upper, unit._threshold))
    return projection_units


def filter_candidate_units(candidate_units, dense_units_smaller_dimension):
    for unit in candidate_units:
        for projection_of_unit in projection(unit):
            if projection_of_unit not in dense_units_smaller_dimension:
                continue
        yield unit


def get_all_units_of_dimension(units, dimension):
    units_of_dimension = []
    for unit in units:
        if len(unit.subspace()) == dimension:
            units_of_dimension.append(unit)
    return units_of_dimension


def find_original_unit_in_units(unit, units):
    for original_unit in units:
        if unit == original_unit:
            return original_unit

def group_by_subspace(units):
    units_by_subspace = {}
    for unit in units:
        if tuple(unit.subspace()) in units_by_subspace:
            units_by_subspace[tuple(unit.subspace())].append(unit)
        else:
            units_by_subspace[tuple(unit.subspace())] = [unit]
    return units_by_subspace


def join_pair(pair):
    return join(pair[0], pair[1])


def add_points_to_units(units, points):
    units_by_subspace = group_by_subspace(units)
    for _units in units_by_subspace.values():
        points_temp = list(points)
        [unit.add_points_within_unit(points_temp) for unit in _units]
    return [u for u_list in units_by_subspace.values() for u in u_list]


def find_dense_units_by_subspaces(units, num_dimension, points):
    def is_dense(unit):
        return unit.is_dense()
    units_by_subspaces = {}
    num_total_points = len(points)
    units_of_one_dimension = get_all_units_of_dimension(units, 1)
    units_of_one_dimension = add_points_to_units(units_of_one_dimension, points)
    candidate_units = list(filter(is_dense, units_of_one_dimension))
    units_by_subspaces.update(group_by_subspace(candidate_units))
    for iteration in range(1, num_dimension):
        pairs_candidate_units = combinations(candidate_units, 2)
        pairs_to_join = filter(lambda pair: can_join_units(*pair), pairs_candidate_units)
        new_candidate_units = map(join_pair, pairs_to_join)
        new_candidate_units = filter_candidate_units(new_candidate_units, candidate_units)
        new_candidate_units = add_points_to_units(new_candidate_units, points)
        new_candidate_units = list(filter(is_dense, new_candidate_units))
        dense_units_by_subspace = group_by_subspace(new_candidate_units)
        if len(dense_units_by_subspace) > 200:
            dense_units_by_pruned_subspace = mdl_pruning(dense_units_by_subspace, num_total_points)
            print("Subspaces of dimension={}, Before pruning = {}, After pruning={}".format(
                iteration+1, len(dense_units_by_subspace), len(dense_units_by_pruned_subspace)))
            units_by_subspaces.update(dense_units_by_pruned_subspace)
            new_candidate_units = reduce(add, dense_units_by_pruned_subspace.values())
        else:
            print("Subspaces of dimension={}, {}".format(
                iteration+1, len(dense_units_by_subspace)))
            units_by_subspaces.update(dense_units_by_subspace)
        candidate_units = list(new_candidate_units)
    return units_by_subspaces


def get_connected_units(dense_units, units, starting_cluster_number, unit_length):
    def neighbour(direction, unit, dimension):
        lower = list(unit._lower)
        upper = list(unit._upper)
        lower[dimension-1] = round(lower[dimension-1] + (unit_length if direction == "right" else -1*unit_length), 1)
        upper[dimension-1] = round(upper[dimension-1] + (unit_length if direction == "right" else -1*unit_length), 1)
        return find_original_unit_in_units(Unit(lower, upper, unit._threshold), dense_units)

    def any_not_visited(units):
        for unit in units:
            if unit.cluster_number == "Undefined":
                return unit

    def depth_first_search(unit, cluster_number):
        #print("cluster {} : {}".format(cluster_number, unit))
        unit.cluster_number = cluster_number
        unit_subspace = unit.subspace()
        for i, dimension in enumerate(unit_subspace):
            left_neighbour = neighbour("left", unit, dimension)
            if ((left_neighbour is not None) and
                    (left_neighbour in dense_units) and
                    (left_neighbour.cluster_number == "Undefined")):
                depth_first_search(left_neighbour, cluster_number)
            right_neighbour = neighbour("right", unit, dimension)
            if ((right_neighbour is not None) and
                    (right_neighbour in dense_units) and
                    (right_neighbour.cluster_number == "Undefined")):
                depth_first_search(right_neighbour, cluster_number)

    not_visited_unit = any_not_visited(dense_units)
    current_cluster_number = starting_cluster_number
    while not_visited_unit:
        depth_first_search(not_visited_unit, current_cluster_number)
        not_visited_unit = any_not_visited(dense_units)
        current_cluster_number += 1
    clusters = {}
    for cluster_number in range(starting_cluster_number, current_cluster_number+1):
        units_in_cluster = list(filter(lambda unit: unit.cluster_number == cluster_number, dense_units))
        if len(units_in_cluster) > 0:
            clusters[cluster_number] = units_in_cluster

    return clusters, current_cluster_number


def mdl_pruning(dense_units_by_subspace, num_total_points):
    @lru_cache(maxsize=None)
    def coverage(subspace):
        dense_units_subspace = (unit for unit in dense_units_by_subspace[subspace] if unit.is_dense())
        return 100. * np.sum((len(unit) for unit in dense_units_subspace))/float(num_total_points)

    @lru_cache(maxsize=None)
    def mean_coverage(tuple_of_subspaces):
        coverages = [coverage(subspace) for subspace in tuple_of_subspaces]
        return np.ceil(np.mean(coverages))

    @lru_cache(maxsize=None)
    def deviation(tuple_of_subspaces):
        return np.ceil(np.fabs([coverage(subspace)-mean_coverage(tuple_of_subspaces) for subspace in tuple_of_subspaces])).tolist()

    def code_length(tuple_of_subspaces):
        if len(tuple_of_subspaces) == 0:
            return 0
        return np.sum(np.log2([mean_coverage(tuple_of_subspaces)] + deviation(tuple_of_subspaces)))

    subspaces = dense_units_by_subspace.keys()
    if len(subspaces) <= 2:
        return dense_units_by_subspace
    sorted_subspaces = sorted(subspaces, key=coverage, reverse=True)
    encoding_lengths = []
    for cut_point in range(len(sorted_subspaces)):
        # don't prune all subspaces
        if cut_point == 0:
            encoding_lengths.append(np.nan)
            continue
        selected_subspaces = tuple(sorted_subspaces[:cut_point])
        pruned_subspaces = tuple(sorted_subspaces[cut_point:])
        encoding_lengths.append(code_length(selected_subspaces) + code_length(pruned_subspaces))
    min_encoding_length_cut_point = np.nanargmin(encoding_lengths)
    selected_subspaces = sorted_subspaces[:min_encoding_length_cut_point]
    return {subspace: dense_units_by_subspace[subspace] for subspace in selected_subspaces}


def clique(points, unit_length=0.1, selectivity=0.1, num_dimension=3, find_clusters_in_only_largest_subspace=True):
    threshold = int(selectivity * len(points))
    subspaces = get_subspaces(num_dimension)
    one_dimensional_subspaces = subspaces[0]
    units = []
    for subspace in one_dimensional_subspaces:
        units.extend(get_units(subspace, num_dimension, unit_length, threshold))
    dense_units_by_subspaces = find_dense_units_by_subspaces(units, num_dimension, points)
    starting_cluster_number = 1
    connected_units_by_subspace = {}
    for subspace, dense_units in dense_units_by_subspaces.items():
        if find_clusters_in_only_largest_subspace:
            largest_subspace = subspaces[-1][0]
            if subspace != largest_subspace:
                continue
        clusters_in_subspace, starting_cluster_number = get_connected_units(dense_units, units,
                                                                            starting_cluster_number, unit_length)
        connected_units_by_subspace[subspace] = clusters_in_subspace
    return connected_units_by_subspace


if __name__ == "__main__":
    import time
    t_start = time.time()
    num_dimension = 10
    actual_num_clusters = 3
    total_points = 1000
    points = random_points(total_points, num_dimension, actual_num_clusters)
    connected_units = clique(points, unit_length=0.1, selectivity=0.005, num_dimension=num_dimension)
    t_end = time.time()
    print(list(connected_units.values())[0].keys())
    print("runtime: {}".format(t_end - t_start))
