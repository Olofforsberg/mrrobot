from __future__ import print_function, unicode_literals, division

import numpy as np

try:
    from numba import njit
except:
    njit = lambda **kw: lambda f: f
    use_jit = False
else:
    use_jit = True


class Blocked(StandardError): pass


blocked_threshold = 0.9


def heappush(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)


def heappop(heap):
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
    else:
        returnitem = lastelt
    return returnitem


@njit(nogil=True)
def cmp_lt(a, b):
    (a1, a2, a3, a4) = a
    (b1, b2, b3, b4) = b
    if a1 >= b1:
        return False
        if a2 >= b2:
            return False
            if a3 >= b3:
                return False
                if a4 >= b4:
                    return False
    return True


@njit(nogil=True)
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if cmp_lt(newitem, parent):
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


@njit(nogil=True)
def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2*pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not cmp_lt(heap[childpos], heap[rightpos]):
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


if not use_jit:
    from heapq import heappush, heappop


@njit(nogil=True)
def manhattan(x, y):
    x1, y1 = x
    x2, y2 = y
    return np.abs(x1 - x2) + np.abs(y1 - y2)


def shortest(grid, heur, source, target, threshold=blocked_threshold, target_size=4):
    "Perform A* search on grid with *source* and target."
    # Basically ripped from NetworkX, thanks!
    source, target = (source[0], source[1]), (target[0], target[1])
    #if grid[target[::-1]]:
    #    raise Blocked('target cell is occupied')
    h, w = grid.shape
    queue = [(0, source, 0, (np.nan, np.nan))]*2
    queue_map = {}
    parent_map = {}
    while len(queue) > 1:
        _, cur, curcost, parent = heappop(queue)
        cx, cy = cur

        if np.all(cur == target):
            path = [target]
            while parent != (np.nan, np.nan):
                path.append(parent)
                parent = parent_map[parent]
            return path[::-1]

        if cur in parent_map:
            continue

        parent_map[cur] = parent

        for dx, dy in ((-1, -1), (0, -1), (1, -1),
                       (-1,  0), (0,  0), (1,  0),
                       (-1,  1), (0,  1), (1,  1)):
            nx, ny = cx + dx, cy + dy
            n = nx, ny
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            elif manhattan(n, target) > target_size and grid[ny, nx] > threshold:
                continue
            elif n in parent_map:
                continue
            ncost = curcost + 1
            if n in queue_map:
                qcost, nheur = queue_map[n]
                if qcost < ncost:
                    continue
            else:
                nheur = manhattan(n, target) + heur[ny, nx]
            queue_map[n] = ncost, nheur
            heappush(queue, (ncost + nheur, n, ncost, cur))

    raise Blocked('a* failed')


def smallest_removal_error(path):
    # Let a, b and c be vectors. We want to calculate the distance d of b from
    # the line passing through a and c by projecting b-a onto the the normal
    # vector of that line.
    b_a = path[1:-1, :] - path[:-2, :]
    c_a = path[2:, :] - path[:-2, :]
    normals = c_a.dot(((0, 1), (-1, 0)))
    normals /= np.sqrt((normals**2).sum())
    errors = np.abs((b_a*normals).sum(axis=1))
    i = errors.argmin()
    return errors[i], i + 1


def smooth(path, max_error):
    path = np.array(path, dtype=float)
    while len(path) > 2:
        smallest, i = smallest_removal_error(path)
        if smallest >= max_error:
            break
        path = np.r_[path[:i, :], path[i+1:, :]]
    return list(path)
