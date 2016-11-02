# -*- coding: utf-8 -*-
'''
'''
from __future__ import print_function
from __future__ import division

import numpy as np
np.seterr(all='raise')
import cv2


def build_system_with_six_unknowns(points_a_, points_b_):
    a = []
    b = []
    for point_a, point_b in zip(points_a_, points_b_):
        x, y = point_a
        x_, y_ = point_b
        a.append([x, y, 0, 0, 1, 0])
        a.append([0, 0, x, y, 0, 1])
        b.append(x_)
        b.append(y_)
    return a, b


def calculate_affine_transformation(src, dst, idxs):
    list_a = [src[idx] for idx in idxs]
    list_b = [dst[idx] for idx in idxs]
    a, b = build_system_with_six_unknowns(list_a, list_b)
    x = np.linalg.solve(a, b)
    return x


def calculate_distance(src, dst, model):
    H = model[:4].reshape(2, -1)
    t = model[-2:]
    d1 = np.linalg.norm((np.dot(H, src) + t) - dst)
    # d2 = np.linalg.norm(src - (np.dot(np.linalg.inv(H), dst) - t))
    # d = d1 + d2
    return d1


def homemade_ransac(data, n, k, t, d, random_state=None):
    """
    Given:
    data – a set of observed data points
    model – a model that can be fitted to data points
    n – the minimum number of data values required to fit the model
    k – the maximum number of iterations allowed in the algorithm
    t – a threshold value for determining when a data point fits a model
    d – the number of close data values required to assert that a model fits well to data

Return:
    bestfit – model parameters which best fit the data (or nul if no good model is found)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    src, dst = data
    iterations = 0
    bestfit = None
    besterr = int(1e5)  # something really large
    while iterations < k:
        maybeinliers_idxs = random_state.choice(len(src), n, replace=False)  # n randomly selected values from data
        maybemodel = calculate_affine_transformation(src, dst, maybeinliers_idxs)
        alsoinliers_idx = set()
        for idx, point in enumerate(src):
            # if idx in maybeinliers_idxs:
            #     continue
            if calculate_distance(src[idx], dst[idx], maybemodel) < t:
                 alsoinliers_idx.add(idx)
        if len(alsoinliers_idx) > d:
            # this implies that we may have found a good model
            # now test how good it is
            bettermodel = maybemodel  # model parameters fitted to all points in maybeinliers and alsoinliers
            thiserr = len(src) - len(alsoinliers_idx)  # a measure of how well model fits these points
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
        iterations += 1
    return bestfit, alsoinliers_idx
