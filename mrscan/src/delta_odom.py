import logging
import threading
import numpy as np
from bisect import bisect_right

import rospy
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry


log = logging.getLogger(__name__)


class DeltaOdometry(object):
    def __init__(self, odom_topic, child=None):
        now = rospy.Time.now()
        self.t,  self.x,  self.y,  self.theta  = now, 0.0, 0.0, 0.0
        self.t0, self.x0, self.y0, self.theta0 = now, 0.0, 0.0, 0.0
        self._previous = [(now, 0.0, 0.0, 0.0)]
        self.sub = rospy.Subscriber(odom_topic, Odometry, self._odom, queue_size=5)
        self.child = child
        self._lock = threading.Lock()

    def _odom(self, msg):
        #log.info('%s: received odometry', self.sub.name)
        with self._lock:
            self._set_pose(msg.header.stamp, msg.pose.pose)

    def _set_pose(self, t, pose):
        pos = pose.position
        rot = pose.orientation
        _, _, theta = euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self._set(t, pos.x, pos.y, theta)

    def _set(self, t, x, y, theta):
        self._previous.append((t, x, y, theta))
        self.t, self.x, self.y, self.theta = t, x, y, theta
        if self._is_unset:
            self._reset()
        if self.child is not None:
            self.child.reset(t)

    @property
    def _is_unset(self):
        return (self.x0, self.y0, self.theta0) == (0.0, 0.0, 0.0)

    def _reset(self, t=None):
        i = -1 if t is None else bisect_right(self._previous, (t,))
        if i >= len(self._previous):
            log.warn('reset into the future')
            i = -1
        #log.info('%s: reset to i = %d (len=%d)', self.sub.name, i, len(self._previous))
        self.t0, self.x0, self.y0, self.theta0 = self._previous[i]
        self._previous = self._previous[i:]

    def _read(self, t=None):
        i = -1 if t is None else bisect_right(self._previous, (t,))
        #log.info('%s: read to i = %d (len=%d)', self.sub.name, i, len(self._previous))
        if i >= len(self._previous):
            raise IndexError('time is in the future')
        t, x, y, theta = self._previous[i]
        dt = t - self.t0
        dtheta = theta - self.theta0
        R = np.array([(np.cos(-self.theta0), -np.sin(-self.theta0)),
                      (np.sin(-self.theta0),  np.cos(-self.theta0))])
        dp = R.dot((x-self.x0, y-self.y0))
        return dt, dp, dtheta

    def is_zero(self):
        with self._lock:
            return self.t == self.t0

    def reset(self, t=None):
        with self._lock:
            self._reset(t)
        if self.child is not None:
            self.child.reset(t)

    def read(self):
        with self._lock:
            dt, dp, dtheta = self._read()
        if self.child is not None:
            dtc, dpc, dthetac = self.child.read()
            dt += dtc
            R = np.array([(np.cos(dtheta), -np.sin(dtheta)),
                          (np.sin(dtheta),  np.cos(dtheta))])
            dp += R.dot(dpc)
            dtheta += dthetac
        return dt, dp, dtheta

    def read_absolute(self):
        with self._lock:
            t, p, theta = self.t, np.r_[self.x, self.y], self.theta
        if self.child is not None:
            dtc, dpc, dthetac = self.child.read()
            t += dtc
            R = np.array([(np.cos(theta), -np.sin(theta)),
                          (np.sin(theta),  np.cos(theta))])
            p += R.dot(dpc)
            theta += dthetac
        return t, p, theta

    def read_reset(self, t=None):
        with self._lock:
            rv = self._read(t=t)
            self._reset(t=t)
            return rv
