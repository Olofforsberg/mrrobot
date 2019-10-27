from __future__ import print_function, unicode_literals, division

import numpy as np
from scipy.ndimage import filters


UNIT = 2e-2  # (m/u)


def gu(v):
    "si units -> grid units"
    return (np.array(v)/UNIT - 0.4999).astype(np.int)


def si(v):
    "grid units -> si units"
    return (np.array(v) + 0.5)*UNIT


class OccupancyGrid(np.ndarray):
    def widen_circle(self, r):
        "Like widen_rect but for circle of radius *r*."
        rs = np.arange(-r, r + 1)
        xs, ys = np.meshgrid(rs, rs)
        kernel = xs**2 + ys**2 <= r**2
        gconv = filters.convolve(self, kernel, mode='nearest')
        return (gconv > 0).astype(self.dtype).view(type(self))

    def blur(self, sigma):
        g = self.astype(np.float64)
        gblur = filters.gaussian_filter(g, sigma, mode='nearest')
        return gblur.view(type(self))

    def is_rect_occupied(self, (x, y), (w, h)):
        "Check if rect with center x, y and sides w, h is occupied in *grid*."
        xs = x + np.arange(w) - w//2
        ys = y + np.arange(h) - h//2
        rows, cols = self.shape
        return np.any(self[np.ix_(np.clip(ys, 0, rows-1),
                                  np.clip(xs, 0, cols-1))])

    def line(self, (x0, y0, x1, y1), weight=1):
        p0, p1 = np.r_[x0, y0], np.r_[x1, y1]
        dp = p1 - p0
        l = np.sqrt(dp.dot(dp))
        dp /= l
        h = UNIT/5.0
        ts = np.arange(0, l+h, h)
        ps = p0 + ts[:, None]*dp[None, :]
        pus = gu(ps)
        self[pus[:, 1], pus[:, 0]] = weight
        #self[gu(ys), gu(xs)] = weight


def loadtxt(f, cls=OccupancyGrid):  #, wall_width=3e-2, wall_sigma=1.0):
    lines = np.loadtxt(f)
    if len(lines) == 0:
        return np.zeros((100, 100), dtype=np.uint8).view(cls)
    x0min, y0min, x1min, y1min = np.amin(lines, axis=0)
    x0max, y0max, x1max, y1max = np.amax(lines, axis=0)
    xmin, ymin = min(x0min, x1min), min(y0min, y1min)
    xmax, ymax = min(x0max, x1max), min(y0max, y1max)
    w, h = xmax - xmin, ymax - ymin
    grid = np.zeros(gu((h, w)) + (1, 1), dtype=np.uint8).view(cls)
    for line in lines:
        grid.line(line)
    #grid = grid.blur(wall_sigma)
    #grid = grid.widen(*gu((wall_width, wall_width)))
    return grid
