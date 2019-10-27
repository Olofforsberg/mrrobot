from __future__ import print_function, unicode_literals, division

import time
import logging

import tf
import rospy
import numpy as np
from std_msgs.msg import Bool
from nav_msgs.msg import Path, OccupancyGrid
from ras_msgs.msg import RAS_Evidence
from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion, Twist, Vector3
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from . import grids, base


tau = 2.0*np.pi
log = logging.getLogger(__name__)


class Robot(base.Robot):
    vision_investigate_svc = None
    arm_svc = None

    def __init__(self, *a, **k):
        rospy.init_node(__package__)
        logging.getLogger('rospy').setLevel(logging.INFO)
        super(Robot, self).__init__(*a, **k)
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.pub_motor = rospy.Publisher('/motor_controller/twist', Twist, queue_size=1)
        self.pub_path = rospy.Publisher('/brain/planned_path', Path, queue_size=5)
        self.pub_grid_occ = rospy.Publisher('/brain/grid/occ', OccupancyGrid, queue_size=5)
        self.pub_grid_heur = rospy.Publisher('/brain/grid/heur', OccupancyGrid, queue_size=5)
        self.pub_pose = rospy.Publisher('/brain/pose', PoseStamped, queue_size=5)
        self.pub_plan_target = rospy.Publisher('/brain/path_target', PointStamped, queue_size=10)
        self.sub_map = rospy.Subscriber('/map/estimate', OccupancyGrid, self._map_update, queue_size=1)
        self.sub_vision = rospy.Subscriber('/brain/mreyes_call/bool', Bool, self._vision_motor_cb, queue_size=10)
        self.sub_vision_target = rospy.Subscriber('/vis/point', PointStamped, self._vision_target_cb, queue_size=10)
        self.sub_evidence = rospy.Subscriber('/evidence', RAS_Evidence, self._evidence_cb, queue_size=10)
        self.sub_goals = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self._goal_cb, queue_size=10)
        self.seen_objects = []
        self.positions = []
        self._is_vision_in_control = False

    def _vision_motor_cb(self, msg):
        log.debug('vision set motor control to %s', msg.data)
        self._is_vision_in_control = msg.data

    def _vision_target_cb(self, msg):
        p = msg.point
        n = np.r_[p.x, p.y]
        #if self.vision_target is None or np.linalg.norm(self.vision_target - n) < 1e-1:
        #    log.debug('vision set target %s', self.vision_target)
        self.vision_target = n

    def _evidence_cb(self, msg):
        self._is_vision_in_control = False
        self.seen_objects.append((msg.object_id, msg.object_location))

    def _goal_cb(self, msg):
        msg = self.tf_listener.transformPose('map', msg)
        pos, rot = msg.pose.position, msg.pose.orientation
        roll, pitch, yaw = euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.goal = (pos.x, pos.y, yaw)

    def plan_path(self, target, *args, **kwds):
        tx, ty = target
        point = PointStamped()
        point.header.frame_id = 'map'
        point.header.stamp = rospy.Time.now()
        point.point.x = tx
        point.point.y = ty
        point.point.z = 0.0
        self.pub_plan_target.publish(point)
        return super(Robot, self).plan_path(target, *args, **kwds)

    def set_motors(self, v, w):
        #log.info('sending %r, %r', v, w)
        if self._is_vision_in_control:
            log.error('refusing to send motor commands when vision '
                      'has control!')
            return
        self.pub_motor.publish(Twist(linear=Vector3(v, 0.0, 0.0),
                                     angular=Vector3(0.0, 0.0, w)))

    def loop(self, it):
        rate = rospy.Rate(self.control_frequency)
        T = 1.0/self.control_frequency
        it = self._publish_grid(self._publish_path(self._update_pos(it)))
        while not rospy.is_shutdown():
            t0 = time.time()
            try:
                next(it)
            except StopIteration:
                rospy.signal_shutdown('job done')
            #dt = time.time() - t0
            #if dt < T:
            rate.sleep()
            dt = time.time() - t0
            if dt > 1.1*T:
                log.warn('task too slow, running below set control '
                         'rate (%.0f%% slow)', (dt/T - 1)*100)

    def investigate(self):
        if self.vision_investigate_svc is None:
            from mreyes.srv import Vision_drive
            self.vision_investigate_svc = rospy.ServiceProxy('vision_investigate', Vision_drive)
        try:
            resp = self.vision_investigate_svc()
            pos = (resp.object_x, resp.object_y, resp.object_z)
            thing = base.Thing(resp.the_object_id, pos)
            return {thing.object_id: thing}
        except:
            log.exception('investigate failed!')
            return {}

    def pick_up(self, thing):
        log.info('picking up %r', thing)
        if self.arm_svc is None:
            from mrarm.srv import armtest
            self.arm_svc = rospy.ServiceProxy('arm_take', armtest)
        self.arm_svc('take')

    def put_down(self):
        log.info('putting thing down')
        if self.arm_svc is None:
            from mrarm.srv import armtest
            self.arm_svc = rospy.ServiceProxy('arm_take', armtest)
        self.arm_svc('put_down')


    def _update_pos(self, it):
        for v in it:
            if self.tf_listener.canTransform('map', 'base', rospy.Time(0)):
                (x, y, z), rot = self.tf_listener.lookupTransform('map', 'base', rospy.Time(0))
                roll, pitch, yaw = euler_from_quaternion(rot)
                self.position = np.r_[x, y]
                self.rotation = yaw
                #self._publish_pose()
                #log.info('%s, %s', self._pf_pose, self._odom_pose)
            else:
                log.error('can\'t update pose, no transform')
            yield v

    def _map_update(self, msg):
        h, w = self.grid.shape
        assert np.isclose(msg.info.resolution, grids.UNIT)
        assert msg.info.width == w and msg.info.height == h
        assert np.isclose(msg.info.origin.position.x, -self.grid.x0)
        assert np.isclose(msg.info.origin.position.y, -self.grid.y0)
        new = (np.r_[msg.data].reshape((h, w)) - 1)/97.0
        if np.any(new != self.grid):
            new = new.view(grids.OccupancyGrid)
            new.x0, new.y0 = self.grid.x0, self.grid.y0
            self.grid = new
            self.update_grid()

    def _publish_pose(self):
        x, y = self.position
        rot = quaternion_from_euler(0, 0, self.rotation)
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = Quaternion(*rot)
        self.pub_pose.publish(pose)

    def _publish_path(self, it):
        for v in it:
            yield v
            if self.current_path is None:
                continue
            p = Path()
            p.header.frame_id = 'map'
            p.header.stamp = rospy.Time.now()
            for tx, ty in self.current_path:
                pose = PoseStamped()
                pose.header = p.header
                pose.pose.position.x = tx
                pose.pose.position.y = ty
                pose.pose.position.z = 0
                p.poses.append(pose)
            self.pub_path.publish(p)

    def _publish_grid(self, it, interval=1.0):
        h, w = self.grid.shape
        msg_occ, msg_heur = OccupancyGrid(), OccupancyGrid()
        msg_occ.header.frame_id = b'map'
        msg_occ.info.resolution = grids.UNIT
        msg_occ.info.height, msg_occ.info.width = h, w
        msg_heur.header, msg_heur.info = msg_occ.header, msg_occ.info
        msg_occ.info.origin.position.x = -self.grid.x0
        msg_occ.info.origin.position.y = -self.grid.y0
        msg_heur.info.origin = msg_occ.info.origin
        msg_occ.data = np.ndarray((h*w,))
        msg_heur.data = np.ndarray((h*w,))
        last_publish = rospy.Time(0)
        for v in it:
            yield v
            now = rospy.Time.now()
            if now - last_publish > rospy.Duration(interval):
                msg_occ.header.stamp = rospy.Time.now()
                msg_heur.header.stamp = rospy.Time.now()
                msg_occ.data[:] = 100*self._grid_occ.reshape((h*w,))
                self.pub_grid_occ.publish(msg_occ)
                last_publish = now
