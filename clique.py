from itertools import (product, combinations, chain)

import numpy as np
import matplotlib.pyplot as pl
from generate_data import random_points


class Unit:
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper
        self._points = []

    def is_point_in_unit(self, point):
        coord_in_range = [((self._lower[coord_index] is None) and True) or ((coord >= self._lower[coord_index]) and (coord < self._upper[coord_index]))
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
                self._points.append(point)

    def is_dense(self, threshold):
        if len(self._points) >= threshold:
            return True

    def __str__(self):
        return "Unit: Lower = {}, Upper = {}, len={}".format(self._lower,
                                                             self._upper,
                                                             len(self))

    def __contains__(self, item):
        return item in self._points

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def __eq__(self, other):
        return (self._lower == other._lower) and (self._upper == other._upper)

    def plot(self):
        x = [self._lower[0], self._upper[0], self._upper[0], self._lower[0], self._lower[0]]
        y = [self._lower[1], self._lower[1], self._upper[1], self._upper[1], self._lower[1]]
        pl.plot(x, y, color='k')


def partition(space, unit_length):
    # space : [[lower values], [upper values]]
    # space = [[0,0], [1,1]]
    # 0 <= unit_length < 1
    intervals_list = []
    for lower, upper in zip(*space):
        if (lower is None) and (upper is None):
            intervals_list.append([None])
            continue
        marks = np.round(np.arange(lower, upper, unit_length), 2).tolist()
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
            if interval is None:
                lower.append(None)
                upper.append(None)
            else:
                lower.append(interval[0])
                upper.append(interval[1])
        if all([(((l is None) and (u is None) and True)
                 or (l < u)) for l, u in zip(lower, upper)]):
            units.append(Unit(lower, upper))
    return units


def is_dense(unit):
    return unit.is_dense(5)


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


def get_units(subspace, num_dimensions, unit_length):
    lower = []
    upper = []
    for i in range(1, num_dimensions+1):
        if i in subspace:
            lower.append(0)
            upper.append(1)
        else:
            lower.append(None)
            upper.append(None)
    return partition([lower, upper], unit_length)


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
    return Unit(lower, upper)


def join_2(unit1, unit2):
    lower = list(unit1._lower[:-1])
    upper = list(unit1._upper[:-1])
    unit1_subspace = unit1.subspace()
    unit2_subspace = unit2.subspace()
    if unit1_subspace[-1] < unit2_subspace[-1]:
        lower.extend([unit1._lower[-1], unit2._lower[-1]])
        upper.extend([unit1._upper[-1], unit2._upper[-1]])
    else:
        lower.extend([unit2._lower[-1], unit1._lower[-1]])
        upper.extend([unit2._upper[-1], unit1._upper[-1]])
    return Unit(lower, upper)


def projection(unit):
    unit_subspace = unit.subspace()
    dimension_minus_one = len(unit_subspace) - 1
    subspaces_minus_one_dimension = combinations(unit_subspace, dimension_minus_one)
    projection_units = []
    for subspace in subspaces_minus_one_dimension:
        lower = [unit._lower[i-1] for i in subspace]
        upper = [unit._upper[i-1] for i in subspace]
        projection_units.append(Unit(lower, upper))
    return projection_units


def filter_candidate_units(candidate_units, dense_units_smaller_dimension):
    units_satisfying_monotonicity = []
    for unit in candidate_units:
        for projection_of_unit in projection(unit):
            if projection_of_unit not in dense_units_smaller_dimension:
                continue
        units_satisfying_monotonicity.append(unit)
    return units_satisfying_monotonicity


def run1():
    points = random_points(200, 2, 3)
    units = partition([[0, 0], [1, 1]], 0.1)
    for u in units:
        u.add_points_within_unit(points)
    dense_units = list(filter(is_dense, units))
    for d in dense_units:
        print(d)
    # plot_dense_units(dense_units)
    # plot_not_dense_points(points, dense_units)
    # pl.show()


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


def clique():
    unit_length = 0.1
    num_dimension = 3
    points = random_points(200, num_dimension, 3)
    subspaces = get_subspaces(num_dimension)
    units = []
    for subspaces_per_dimension in subspaces:
        for subspace in subspaces_per_dimension:
            units.extend(get_units(subspace, num_dimension, unit_length))
    for unit in units:
        unit.add_points_within_unit(points)
    units_of_fixed_dimension = get_all_units_of_dimension(units, 1)
    candidate_units = list(filter(is_dense, units_of_fixed_dimension))
    all_dense_units = list(candidate_units)
    for iteration in range(1, num_dimension):
        pairs_candidate_units = combinations(candidate_units, 2)
        new_candidate_units = []
        for unit1, unit2 in pairs_candidate_units:
            if can_join_units(unit1, unit2):
                joined_unit = join(unit1, unit2)
                original_unit = find_original_unit_in_units(joined_unit, units)
                new_candidate_units.append(original_unit)
        new_candidate_units = list(filter(is_dense, new_candidate_units))
        new_candidate_units = filter_candidate_units(new_candidate_units, candidate_units)
        all_dense_units.extend(new_candidate_units)
        candidate_units = new_candidate_units
    units_by_subspaces = group_by_subspace(all_dense_units)
    print(units_by_subspaces.keys())




if __name__ == "__main__":
    clique()
