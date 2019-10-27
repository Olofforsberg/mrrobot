from __future__ import print_function, unicode_literals, division

import numpy as np
from scipy.ndimage import filters


def gu(v):
    "si units -> grid units"
    return (np.array(v)/UNIT - 0.4999).astype(np.int)


def si(v):
    "grid units -> si units"
    return (np.array(v) + 0.5)*UNIT


UNIT = 2e-2  # (m/u)
SIZE = 5.5, 3.0
SIZE_UNITS = gu(SIZE)


class OccupancyGrid(np.ndarray):
    def widen_circle(self, r):
        "Widen for circle of radius *r*."
        rs = np.arange(-r, r + 1)
        xs, ys = np.meshgrid(rs, rs)
        kernel = xs**2 + ys**2 <= r**2
        gconv = filters.convolve(self, kernel, mode='nearest')
        new = np.fmin(1.0, gconv.astype(self.dtype).view(type(self)))
        if hasattr(self, 'x0'):
            new.x0, new.y0 = self.x0, self.y0
        return new

    def blur(self, sigma):
        g = self.astype(np.float64)
        gblur = filters.gaussian_filter(g, sigma, mode='nearest')
        new = gblur.view(type(self))
        if hasattr(self, 'x0'):
            new.x0, new.y0 = self.x0, self.y0
        return new

    def _line_idxs(self, (p0x, p0y), (p1x, p1y), width=0.03):
        h, w = self.shape
        xs, ys = np.meshgrid(si(np.arange(w)) - p0x,
                             si(np.arange(h)) - p0y)
        vx, vy = float(p0x - p1x), float(p0y - p1y)
        vlen2 = vx**2 + vy**2
        nx, ny = vy, -vx
        s = (nx*xs + ny*ys) / np.sqrt(vlen2)
        t = -(vx*xs + vy*ys)
        return (np.abs(s) < width)*(0 < t)*(t < vlen2)

    def line(self, (p0x, p0y, p1x, p1y), weight=1, width=0.03):
        self[self._line_idxs((p0x, p0y), (p1x, p1y), width=width)] = weight

    def line_max(self, p0, p1, width=0.03):
        return np.max(self[self._line_idxs(p0, p1, width=width)])

    def line_sum(self, p0, p1, width=0.03):
        return np.sum(self[self._line_idxs(p0, p1, width=width)])

def loadtxt(f, cls=OccupancyGrid, dtype=np.uint8):
    grid = np.zeros(SIZE_UNITS, dtype=dtype).view(cls)
    h, w = SIZE
    lines = np.loadtxt(f)
    if len(lines) > 0:
        x0min, y0min, x1min, y1min = np.amin(lines, axis=0)
        x0max, y0max, x1max, y1max = np.amax(lines, axis=0)
        xmin, ymin = min(x0min, x1min), min(y0min, y1min)
        xmax, ymax = max(x0max, x1max), max(y0max, y1max)
        wmap, hmap = xmax - xmin, ymax - ymin        
        assert wmap <= w and hmap <= h
        x0, y0 = (w-wmap)/2.0, (h-hmap)/2.0
        for line in lines:
            grid.line(line + (x0, y0, x0, y0))
    else:
        x0, y0 = w/2.0, h/2.0
    grid.x0, grid.y0 = x0, y0
    return grid
