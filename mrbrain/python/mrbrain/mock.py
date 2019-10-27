from __future__ import print_function, unicode_literals, division

import time
import logging
import numpy as np

from . import base


tau = 2.0*np.pi
log = logging.getLogger(__name__)


class Robot(base.Robot):
    _is_running = True

    def set_motors(self, v, w):
        super(Robot, self).set_motors(v, w)
        self.motors = v, w

    def loop(self, it):
        T = 1.0/self.control_frequency
        self.positions = []
        while self._is_running:
            t0 = time.time()
            try:
                next(it)
            except StopIteration:
                self._is_running = False
            dt = time.time() - t0
            if dt < T:
                time.sleep(T - dt)
            elif dt > 1.1*T:
                log.warn('task too slow, running below set control '
                         'rate (%.0f%% slow)', (dt/T - 1.0)*100)
            self._simulate_engines(time.time() - t0)

    def _simulate_engines(self, T):
        self.positions.append(tuple(self.position))
        v, w = self.motors
        r = self.rotation
        #dp = v/w*( (sin(r + w*T) - sin(r),
        #           -cos(r + w*T) - -cos(r)) )
        self.position += v*T*np.r_[np.cos(r), np.sin(r)]
        self.rotation  = (r + w*T) % tau
        #log.debug('robot config: %s', np.r_[self.position, self.rotation])
