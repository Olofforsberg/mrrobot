from __future__ import print_function, unicode_literals, division

import os
import time
import logging
import logcolor
import numpy as np
from os import path
from itertools import chain
from collections import namedtuple

from . import pathing, grids
from .grids import gu, si


tau = 2.0*np.pi
log = logging.getLogger(__name__)


class CantLocate(StandardError):
    pass


Thing = namedtuple('Thing', 'object_id position')


class Robot(object):
    "Holds high-level state and description of robot"

    control_frequency = 10.0 # Hz
    radius = 145e-3 # m
    max_v = 0.27 # m/s
    max_w = 0.15*tau # rad/s
    tol_targets = 5e-2 # m
    tol_vw = tau*30/360/max_v
    break_distance = 1e-1 # m
    lookahead_distance = 8e-2 # m
    wall_avoid_sigma = 1.2
    wall_avoid_scale = 7e1
    max_smoothing_error = 3e-2 # m
    tol_abduction = 5e-1 # m
    max_v_exploring = 0.15 # m/s
    max_v_investigating = 0.10 # m/s

    def __init__(self, position=(0.0, 0.0), rotation=0.0, home=None, grid=None):
        logcolor.basic_config(level=logging.DEBUG)
        self.position = np.r_[position]
        self.home_position = np.r_[position if home is None else home]
        self.rotation = rotation
        self.current_path = None
        self.grid = grid
        self.update_grid()
        with open(path.expanduser('~/mrbrain-things.txt')) as f:
            self.things = [Thing(*eval(line)) for line in f]

    def update_grid(self):
        self._grid_occ = (self.grid > pathing.blocked_threshold).widen_circle(gu(self.radius))
        blurred = self._grid_occ.blur(sigma=self.wall_avoid_sigma)
        self._grid_heur = self.wall_avoid_scale*blurred

    def set_motors(self, v, w):
        #log.info('set motors v=%f, w=%f', v, w)
        pass

    def rotate(self, rot, tol=0.05):
        rerr = float('inf')
        while np.abs(rerr) > tol:
            rerr = ((np.pi + rot - self.rotation) % tau) - np.pi
            w = rerr if abs(rerr) < self.max_v*self.tol_vw else np.sign(rerr)*self.max_w
            w = max(-self.max_w, min(+self.max_w, w))
            self.set_motors(0.0, w)
            yield

    def rotate_to_target(self, target, tol=0.05):
        rerr = float('inf')
        while np.abs(rerr) > tol:
            dx, dy = target - self.position
            rot = np.arctan2(dy, dx)
            rerr = ((np.pi + rot - self.rotation) % tau) - np.pi
            w = rerr if abs(rerr) < self.max_v*self.tol_vw else np.sign(rerr)*self.max_w
            w = max(-self.max_w, min(+self.max_w, w))
            self.set_motors(0.0, w)
            yield

    def move_point(self, target, check_blocked=True):
        log.info('moving towards %s from %s', target, self.position)
        derr = float('inf')
        while derr > self.tol_targets:
            dx, dy = target - self.position
            rot = np.arctan2(dy, dx)
            derr = np.sqrt(dy**2 + dx**2)
            rerr = ((np.pi + rot - self.rotation) % tau) - np.pi

            w = rerr if abs(rerr) < self.max_v*self.tol_vw else np.sign(rerr)*self.max_w
            w = max(-self.max_w, min(+self.max_w, w))
            v = min(derr, self.break_distance)/self.break_distance*self.max_v
            v = v if abs(w) < self.max_w else 0

            next_pos = self.position
            if v > 0:
                next_pos += self.lookahead_distance*np.r_[dx, dy]/derr
            xu, yu = gu(next_pos + (self.grid.x0, self.grid.y0))
            if self._grid_occ[yu, xu] > pathing.blocked_threshold:
                #log.error('we seem to have collided!')
                if check_blocked:
                    raise pathing.Blocked('cell occupied once we got there')
                else:
                    log.warn('collision detected, but asked to ignore :(')

            self.set_motors(v, w)
            yield

    def move_unstuck(self):
        xu, yu = gu(self.position + (self.grid.x0, self.grid.y0))
        v = 5e-2*np.r_[np.cos(self.rotation), np.sin(self.rotation)]
        p = self.position
        ls = self._grid_occ.line_sum
        backout_dir = -1 if ls(p, p+v) >= 0.99*ls(p, p-v) else +1
        while self._grid_occ[yu, xu] > pathing.blocked_threshold:
            self.set_motors(backout_dir*0.1, 0)
            yield
            xu, yu = gu(self.position + (self.grid.x0, self.grid.y0))

    def move_point_unstick(self, target, num_attempts=3):
        last = self.position
        for i in range(num_attempts):
            try:
                for v in self.move_point(target):
                    yield v
                    if ((last - self.position)**2).sum() > self.tol_abduction:
                        raise pathing.Blocked('robot was abducted!')
                    last = self.position
            except pathing.Blocked as e:
                log.warn('unsticking, robot stuck: %s', e)
                for v in self.move_unstuck():
                    yield v
            else:
                break
        else:
            for v in self.move_unstuck():
                yield v
            raise pathing.Blocked('robot keeps getting stuck')

    def move_path(self, path):
        try:
            for target in path[1:]:
                for v in self.move_point_unstick(target):
                    yield v
        finally:
            self.set_motors(0, 0)

    def plan_path(self, target, target_size=5e-2):
        log.info('planning path from %s', self.position)
        grid_origin = np.r_[self.grid.x0, self.grid.y0]
        path = pathing.shortest(self._grid_occ, self._grid_heur,
                                gu(self.position + grid_origin),
                                gu(target + grid_origin),
                                target_size=gu(target_size))
        path = pathing.smooth(path, max_error=self.max_smoothing_error)
        #print(occ.format_with_path(path))
        return si(path + [gu(target + grid_origin)]) - grid_origin

    def move_planned(self, target, target_size=1e-2, num_attempts=5, rotation=None):
        for i in range(num_attempts):
            for v in self.move_unstuck():
                yield v
            log.info('planning path to %r (attempt %d)', target, i+1)
            self.current_path = self.plan_path(target, target_size=target_size)
            try:
                log.info('constructed path:\n%s', self.current_path)
                for v in self.move_path(self.current_path):
                    yield v
                break
            except pathing.Blocked as e:
                log.error('move plan failed, path blocked: %s', e)
        else:
            raise pathing.Blocked('retried {} times, failed'.format(num_attempts))
        if rotation is not None:
            for v in self.rotate(rotation):
                yield v

    def move_goals(self):
        self.goal = None
        while True:
            if self.goal is None:
                yield
                continue
            (x, y, theta), self.goal = self.goal, None
            log.info('moving to goal: %r', (x, y, theta))
            try:
                for v in self.move_planned((x, y), rotation=theta):
                    yield v
            except pathing.Blocked as e:
                log.error('move to goal failed, path blocked: %s', e)
            log.info('waiting for next goal')

    def investigate(self):
        log.info('investigating')
        raise NotImplementedError()

    def pick_up(self, thing):
        log.info('picking up %r', thing)
        raise NotImplementedError()

    def put_down(self):
        log.info('putting thing down')
        raise NotImplementedError()

    def go_home(self):
        return self.move_planned(self.home_position)

    def move_close_to(self, target, distance=1e-1, tol=3e-2):
        diff = self.position - target
        d = np.linalg.norm(diff)
        line_target = self.position - diff*(1 - distance/d)
        grid_origin = np.r_[self.grid.x0, self.grid.y0]
        if np.linalg.norm(self.position - line_target) < tol:
            log.info('move_close_to: already at destination')
            yield
        elif self._grid_occ.line_max(self.position + grid_origin,
                                     line_target + grid_origin) < pathing.blocked_threshold:
            log.info('move_close_to: target is on a clear line path')
            for v in self.move_point(line_target, check_blocked=False):
                yield v
        elif np.linalg.norm(target - self.position) > distance:
            log.info('move_close_to: non-trivial target, planning a path')
            old = self.grid.copy()
            h, w = self.grid.shape
            size2 = 3e-2
            tx, ty = target
            xs, ys = np.meshgrid(si(np.arange(w)) - self.grid.x0,
                                 si(np.arange(h)) - self.grid.y0)
            self.grid[(xs - tx)**2 + (ys - ty)**2 < size2] = 0
            self.update_grid()
            try:
                for v in self.move_planned(target):
                    if np.linalg.norm(target - self.position) <= distance:
                        break
                    yield v
            finally:
                self.grid[:, :] = old
                self.update_grid()
        else:
            log.info('move_close_to: target is already within given distance')
        for v in self.rotate_to_target(target):
            yield v

    def retrieve(self, thing):
        log.info('retrieving %r', thing)
        for v in self.move_close_to(thing.position[:2], distance=30e-2):
            yield v
        for v in self.sleep(0.1):
            yield v
        for v in self.rotate_to_target(thing.position[:2]):
            yield v
        found = self.investigate()
        for v in self.move_unstuck():
            yield v
        if not found:
            raise CantLocate('could not find %r, found nothing' % (thing,))
        elif thing.object_id not in found:
            log.warn('could not find %r, but found %r' % (thing, found))
        os.system('aplay ~/choppa.wav')
        self.pick_up(found.keys()[0])
        for v in self.go_home():
            yield v
        self.put_down()

    def retrieve_all(self):
        shape = lambda t: t.object_id.rsplit(' ', 1)[-1].lower()
        shape_values = {'ball': 10, 'hollow cube': 5,
                        'cross': 1, 'star': 1, 'patric': 1,   
                        'cube': 0.1, 'triangle': 0.1, 'cylinder': 0.1,
                        'object': 0.0}
        shape_confidence = {'ball': 0, 'hollow cube': 0,
                            'cross': 1, 'star': 0, 'patric': 0,
                            'cube': 2, 'triangle': 0, 'cylinder': 1,
                            'object': 0}
        things = list(self.things)
        things.sort(key=lambda t: (shape_values[shape(t)],
                                   shape_confidence[shape(t)],
                                  -np.linalg.norm(t.position[:2] - self.position)))
        log.info('retrieving all things in following order:\n%s',
                 '\n'.join(' - {0.object_id}'.format(t) for t in things))
        while things:
            thing = things.pop()
            try:
                for v in self.retrieve(thing):
                    yield v
            except:
                log.exception('could not retrieve %s', thing)

    def explore_to(self, (x, y), rotation=None,
                   min_distance=27e-2, min_distance_ignored=1e-1,
                   min_distance_investigate=40e-2):
        for v in chain(self.rotate_to_target((x, y)),
                       self.move_close_to((x, y))):
            if self.vision_target is not None:
                if all(np.linalg.norm(self.vision_target - t) > min_distance_ignored
                       for t in self.ignore_targets):
                    break
                else:
                    log.debug('ignoring target %s', self.vision_target)
            yield v
        else:
            return

        old_v, self.max_v = self.max_v, self.max_v_investigating
        self.ignore_targets.append(self.vision_target)

        try:
            log.info('moving to vision target: %s', self.vision_target)
            for w in self.move_close_to(self.vision_target, distance=min_distance):
                yield w
        except pathing.Blocked as e:
            log.error('move to vision target failed: %s', e)
        finally:
            self.max_v = old_v

        if np.linalg.norm(self.vision_target - self.position) < min_distance_investigate:
            for thing in self.investigate().values():
                try:
                    os.system('aplay ~/choppa.wav')
                    self.pick_up(thing)
                    for v in self.go_home():
                        yield v
                    self.put_down()
                except:
                    log.exception('could not retrieve %s', thing)
        else:
            log.warn('did not come close enough to vision target!')
        for v in self.move_unstuck():
            yield v
        self.vision_target = None

    def explore(self):
        self.goal = None
        self.vision_target = None
        self.ignore_targets = []
        while True:
            # This part will come from a service call to exploring node
            if self.goal is None:
                yield
                continue
            (x, y, theta), self.goal = self.goal, None

            log.info('moving to goal: %r', (x, y, theta))
            old_v, self.max_v = self.max_v, self.max_v_exploring
            try:
                for v in self.explore_to((x, y), rotation=theta):
                    yield v
            except pathing.Blocked as e:
                log.error('move to goal failed, path blocked: %s', e)
            finally:
                self.max_v = old_v
            log.info('waiting for next goal')

    def add_thing(self, thing):
        log.info('found thing %s', thing)
        self.things.append(thing)
        with open(path.expanduser('~/mrbrain-things.txt'), 'wb') as f:
            f.write(''.join('%r\n' % (tuple(thing),) for thing in self.things))

    def path_eta(self, path, fudge=0.90):
        "Estimate time it takes to go to target using path"
        deltas = np.diff(path, axis=0)
        dists = np.sqrt((deltas**2).sum(axis=1))
        delta_angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        rots = np.abs(np.diff(np.r_[self.rotation, delta_angles]))
        return (dists.sum()/self.max_v + rots.sum()/self.max_w)/fudge

    def sleep(self, t):
        t0 = time.time()
        while time.time() - t0 < t:
            yield

    def retreating(self, it, timeout, home=None, tol2=(2e-1)**2,
                   eta_error_margin=5.0):
        "Stop executing *it* if it takes too long to get back to home."
        home = home if home is not None else self.home_position
        t0 = time.time()
        return_home_eta = None
        last_eta_pos = float('inf'), float('inf')
        while True:
            yield next(it)
            time_remaining = t0 + timeout - time.time()
            if ((self.position - last_eta_pos)**2).sum() > tol2:
                log.debug('replanning retreat')
                return_home_eta = self.path_eta(self.plan_path(home))
                return_home_eta += eta_error_margin
                last_eta_pos = self.position
                log.debug('updated eta for returning home: %.2fs', return_home_eta)
            if return_home_eta >= time_remaining:
                log.warn('returning home: %.2fs remaining', time_remaining)
                break
        for v in self.move_planned(home):
            yield v

    def visualizing(self, it, dirname='viz', interval=.1):
        from os import path
        t0 = 0
        for i, v in enumerate(it):
            if time.time() - t0 > interval:
                im = self.snapshot(path=self.current_path)
                fn = path.join(dirname, 'f{:05d}.png'.format(i))
                im.save(fn, 'PNG')
                t0 = time.time()
            yield v

    def snapshot(self, path=None):
        from PIL import Image, ImageDraw
        h, w = self.grid.shape
        u = si(100) # pix/u
        im = Image.new('L', (int(w*u+1), int(h*u+1)))
        d = ImageDraw.Draw(im)
        ru = self.radius/grids.UNIT
        xu, yu = self.position/grids.UNIT - 0.4999

        # Cell coloring
        for x in range(w):
            x0, x1 = int(x*u), int((x+1)*u)
            for y in range(h):
                y0, y1 = int(y*u), int((y+1)*u)
                c = 0
                c = 255*self.grid[y, x]
                c = (c, 128)[int(abs(x - xu + 0.5)) < ru and int(abs(y - yu + 0.5)) < ru]
                c = (c, 192)[(x, y) == (int(xu + 0.5), int(yu + 0.5))]
                d.rectangle((x0, y0, x1, y1), c)

        # Grid
        #for x in range(w):
        #    d.line((int(x*u), 0, int(x*u), im.size[1]), 32)
        #for y in range(h):
        #    d.line((0, int(y*u), im.size[0], int(y*u)), 32)

        def draw_path(path, line, point=None, width=2):
            for p in np.c_[path[:-1], path[1:]]/grids.UNIT:
                d.line(tuple((p*u).astype(int)), line, width=2)
            if point is not None:
                for p in path[:-1]/grids.UNIT:
                    im.putpixel(tuple((p*u).astype(int)), point)

        # Path
        if path is not None:
            draw_path(path, line=96, point=255)

        # Positions
        draw_path(self.positions, line=192, width=1)

        # Pose
        r = self.rotation
        p0 = self.position
        p1 = p0 + self.radius*np.r_[np.cos(r), np.sin(r)]
        d.line(tuple((np.r_[p0, p1]/grids.UNIT*u).astype(int)), 192)
        return im
