#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from mrbrain import grids
import sys
import logging, logcolor

from os import path
from contextlib import contextmanager
from timeit import default_timer

import rospy, tf
import numpy as np
from scipy.ndimage import filters
from tf.transformations import euler_from_quaternion,quaternion_from_euler
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from random import uniform

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from move_base_msgs.msg import MoveBaseGoal
from geometry_msgs.msg import PoseStamped,Quaternion


try:
    from numba import jit
except ImportError as e:
    jit = lambda **kw: lambda f: f
    use_jit = False
else:
    use_jit = True

#from . import grids

#coverage mapping psudo code

#main thing:
# use build up map to construct covarage map
# update coverage map given robo pose and field of view
# have costfunction for possible points (Closesd point, information gain, etc)
#print("hello")

log = logging.getLogger('covermapper')

def gu(v):
    "si units -> grid units"
    return (np.array(v)/UNIT - 0.4999).astype(np.int)

def si(v):
    "grid units -> si units"
    return (np.array(v) + 0.5)*UNIT

UNIT = 2e-2  # (m/u)
SIZE = 5.5, 3.0
SIZE_UNITS = gu(SIZE)

ROBO_SIZE=0.17

HIT_UNKNOWN = 0
HIT_FREE = 1
HIT_OCCUPIED = 2
HIT_ROBOT = 4
HIT_VIEW= 8


#Field of vew parameter
VIEW_DISTANCE=0.5
VIEW_ANGLE=1.7/2 # ~45°

SIGMA=0.75
#NUMPOINTS=200


@contextmanager
def timed_code(name=None):
    next_unit = iter(("s", "ms", "ns", "us")).next
    msg = "section %s took" % (name,) if name else "section took"
    t0 = default_timer()
    try:
        yield msg
    finally:
        delta = default_timer() - t0
        unit = next_unit()
        while delta < 1:
            delta *= 1000.0
            try:
                unit = next_unit()
            except StopIteration:
                break
        log.info("%s: %.2f%s" % (msg, delta, unit))

angle_min = -3.12413907051
angle_max = +3.14159274101
angle_increment = 0.0174532923847
a = np.arange(angle_min, angle_max + angle_increment, angle_increment)
angle_error=1.3*np.pi/180 #1/16 degree

p_0 = 0.1
p_view = 0.6
p_robo = 0.8
l_0 = np.log(p_0/(1-p_0))
l_view = np.log(p_view/(1-p_view))
l_robo = np.log(p_robo/(1-p_robo))


angle_from=-np.pi*0.75
angle_to=-np.pi*0.3


@jit(nopython=True, nogil=True)
def beam_map(p0x, p0y, points, prx, pry, theta, step=1):
    "Shoot triangular beams. Pew pew pew."
    assert xs.shape == ys.shape
    hits = np.zeros(xs.shape, dtype=np.uint8)
    for i in range(points.shape[0] - 1, -1 + 1, -step):
        ri, ai = points[i, :]
        rj, aj = points[i-1, :]
        if not ((0.15 < ri < np.inf) and (0.15 < rj < np.inf)):
            continue
        if (ri-rj)**2 > (0.10)**2:
            continue
        px  = points[i, 0]*np.cos(points[i, 1]) + p0x
        py  = points[i, 0]*np.sin(points[i, 1]) + p0y
        qx  = points[i-1, 0]*np.cos(points[i-1, 1]) + p0x
        qy  = points[i-1, 0]*np.sin(points[i-1, 1]) + p0y
        vx, vy = qx - px, qy - py
        vl = np.sqrt(vx**2 + vy**2)
        p1x, p1y = px, py
        p2x, p2y = qx, qy
        dblarea = (-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
        s = 1.0/(dblarea)*(p0y*p2x - p0x*p2y + (p2y - p0y)*xs + (p0x - p2x)*ys)
        t = 1.0/(dblarea)*(p0x*p1y - p0y*p1x + (p0y - p1y)*xs + (p1x - p0x)*ys)
        points_free = (0 <= s)*(s <= 1)*(0 <= t)*(t <= 1)*(0 <= 1 - s - t)
        #vlen = np.sqrt((p2x-p1x)**2 + (p2y - p1y)**2)
        #points_occupied = points_free*(np.abs(dblarea*(1-s-t)) <= 0.025*vlen)
        #hits[points_free] |= HIT_FREE
        idx_rows, idx_cols = np.nonzero(points_free)
        for i in range(idx_rows.shape[0]):
            hits[idx_rows[i], idx_cols[i]] |= HIT_VIEW
    #hits for robo pose
    xt=xs-prx #translate
    yt=ys-pry
    xr   =  np.cos(theta)*xt + np.sin(theta)*yt  # "cloclwise"
    yr   = -np.sin(theta)*xt + np.cos(theta)*yt
    xt=xr+prx #translate back
    yt=yr+pry
    points_free_robo=(np.abs(prx - xt) < ROBO_SIZE/2.0)*(np.abs(pry - yt) < ROBO_SIZE/2.0)
    idx_rows, idx_cols = np.nonzero(points_free_robo)
    for i in range(idx_rows.shape[0]):
        hits[idx_rows[i], idx_cols[i]] |= HIT_ROBOT


    
    return hits


@jit(nopython=True, nogil=True)
def beam_map2(p0x, p0y, theta, step=1,view_distance=VIEW_DISTANCE,view_angle=VIEW_ANGLE):
    "Shoot 1 triangular beams. Pew pew pew."
    assert xs.shape == ys.shape
    hits = np.zeros(xs.shape, dtype=np.uint8)
    
    p1x  = view_distance*np.cos( theta + view_angle) + p0x
    p1y  = view_distance*np.sin( theta + view_angle) + p0y
    p2x  = view_distance*np.cos( theta - view_angle) + p0x
    p2y  = view_distance*np.sin( theta - view_angle) + p0y
    dblarea = (-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
    s = 1.0/(dblarea)*(p0y*p2x - p0x*p2y + (p2y - p0y)*xs + (p0x - p2x)*ys)
    t = 1.0/(dblarea)*(p0x*p1y - p0y*p1x + (p0y - p1y)*xs + (p1x - p0x)*ys)
    points_fieldofview = (0 <= s)*(s <= 1)*(0 <= t)*(t <= 1)*(0 <= 1 - s - t)
    idx_rows, idx_cols = np.nonzero(points_fieldofview)
    for i in range(idx_rows.shape[0]):
        hits[idx_rows[i], idx_cols[i]] |= HIT_VIEW

    return hits



class CoverageMapBuilder(object):
    sigma = 0.2
    min_time = rospy.Duration(0.5)

    def __init__(self,mapfile):
        self.lx=0.0
        self.ly=0.0
        self.ltheta=0
        self.x=0.0
        self.y=0.0
        self.theta=0
        self.occ_map, self.x0, self.y0,self.xmin,self.xmax,self.ymin,self.ymax = loadtxt(mapfile, dtype=np.float64)
        self.explore_map = np.zeros(self.occ_map.shape).view(type(self.occ_map))
        self.log_map = l_0*np.ones(self.occ_map.shape).view(type(self.occ_map))
        h, w = self.log_map.shape            
        xs, ys = np.meshgrid(si(np.arange(w)) - self.x0,
                             si(np.arange(h)) - self.y0)
        self.explore_map[(xs<self.xmin)]=1
        self.explore_map[(xs>self.xmax)]=1
        self.explore_map[(ys<self.ymin)]=1
        self.explore_map[(ys>self.ymax)]=1

        self.tf_listener = tf.TransformListener()
        self.sub_lidar = rospy.Subscriber('/scan', LaserScan, self.lidar_cb, queue_size=1)
        self.pub_point = rospy.Publisher('/map/nextpoint', Marker, queue_size=1)
        self.pub_allpoints = rospy.Publisher('/map/nextpossiblepoints', MarkerArray, queue_size=1)
        self.mb_goal = MoveBaseGoal()
        self.pub_map = rospy.Publisher('/map/coveragemap', OccupancyGrid, queue_size=1)
        self.pub_goal = rospy.Publisher('/map/nextgoal', PoseStamped)
        self.sub_map = rospy.Subscriber('/brain/grid/occ', OccupancyGrid, self._map_update, queue_size=1)
        self.last_update = rospy.Time.now()
    
    
    def _map_update(self, msg):
        log.info('update map')
        h, w = self.occ_map.shape
        assert np.isclose(msg.info.resolution, UNIT)
        assert msg.info.width == w and msg.info.height == h
        assert np.isclose(msg.info.origin.position.x, -self.x0)
        assert np.isclose(msg.info.origin.position.y, -self.y0)
        new = (np.r_[msg.data].reshape((h, w)) - 1)/97.0
        if np.any(new != self.occ_map):
            self.occ_map = new.view(type(self.occ_map))

    def lidar_cb(self,msg):
        if msg.header.stamp - self.last_update < self.min_time:
            return
        
        #read in ranges from lidar only for a given range
        #global r
        r = np.array(msg.ranges, dtype=np.float64)
        
        rsmall = r[(a>=angle_from)*(a<=angle_to)]
        asmall = a[(a>=angle_from)*(a<=angle_to)]
        #a = np.array(msg.angle, dtype=mp.float64)

        t = msg.header.stamp #rospy.Time(0)
        if self.tf_listener.canTransform('map', 'laser', t):
            (self.lx, self.ly, _), rot = self.tf_listener.lookupTransform('map', 'laser', t)
            _, _, self.ltheta = euler_from_quaternion(rot)
            (self.x, self.y, _), rot = self.tf_listener.lookupTransform('map', 'base', t)
            _, _, self.theta = euler_from_quaternion(rot)
            log.info('set pose: %.2f, %.2f, %.2f', self.lx, self.ly, self.ltheta)
        else:
            log.warn('did not find current pose, ignoring update')
            return

        with timed_code('update'):
            h, w = self.log_map.shape
            rsmall[rsmall>VIEW_DISTANCE]=VIEW_DISTANCE

            points = np.c_[rsmall, -asmall + self.ltheta]
            global xs, ys
            xs, ys = np.meshgrid(si(np.arange(w)) - self.x0, si(np.arange(h)) - self.y0)
            with timed_code('-beam map'):
                hits = beam_map(self.lx, self.ly, points, self.x, self.y, self.theta)
            self.log_map[hits > 0]    += (l_view - l_0)
        #translate log map to occupancygrid map (0-100)
        self.explore_map[:, :] = 1.0 - 1.0/(1.0 + np.exp(self.log_map))
        #self.getnewexploregoal(hitslidar)
        self.last_update = msg.header.stamp

        #**********************************************
        # get cool new point *****************************
        hits = np.zeros(xs.shape, dtype=np.uint8)
        #ris = [i for i in range(len(r)) if (0.15 < r[i] < 4)]
        ris = list(range(0, len(r), 2))
        sample = [i for i in ris if r[i] > 0.12]
        #sample = np.random.choice(sample, 100)
        randomr=[uniform(0.12,r[i]-0.05) for i in sample]
        randomx=[self.lx +randomr[j]*np.cos(-a[i]+self.ltheta) for j, i in enumerate(sample)]
        randomy=[self.ly +randomr[j]*np.sin(-a[i]+self.ltheta) for j, i in enumerate(sample)]
        
        costs=np.zeros(len(randomx))
        countfree=0
        for i in range(len(randomx)):
            if((randomx[i]>np.min(xs))*(randomx[i]<np.max(xs))*(randomy[i]>np.min(ys))*(randomy[i]<np.max(ys))):
                costs[i]+=np.sqrt((self.x-randomx[i])**2+(self.y-randomy[i])**2)*1
                angle=(-a[i]+self.ltheta-self.theta)*180/np.pi
                costs[i]+=np.abs((np.mod(angle+180,360)-180))*100

                #info gain
                hits = beam_map2(randomx[i],randomy[i],-a[i]+self.ltheta)
                costs[i]+=np.sum(self.explore_map[hits == HIT_VIEW])*1
                #self.explore_map[hits == HIT_VIEW]=1
                #penelty for point we have been to 
                if(self.explore_map[gu(randomy[i]+self.y0),gu(randomx[i]+self.x0)]>0.2):
                    costs[i]+=1000
                    countfree+=1
                if(self.occ_map[gu(randomy[i]+self.y0),gu(randomx[i]+self.x0)]>0.2):
                    costs[i]+=10e3
                    countfree+=1
                elif(self.occ_map.line_max((self.x, self.y), (randomx[i]+self.x0, randomy[i]+self.y0))>0.2):
                    costs[i]+=1000
                    countfree+=1

            else:                
                costs[i]=np.inf

        #new points as we are in a deadend
        if False and (countfree>len(randomx)-2):
            costs=np.zeros(len(randomx))
            randomx=[uniform(self.xmin,self.xmax) for i in range(0,len(r))]  
            randomy=[uniform(self.ymin,self.ymax) for i in range(0,len(r))]   
            randoma=[uniform(-np.pi,np.pi) for i in range(0,len(r))]   
            
            costs=np.zeros(len(randomx))
            countfree=0
            for i in range(len(randomx)):
                if((randomx[i]>np.min(xs))*(randomx[i]<np.max(xs))*(randomy[i]>np.min(ys))*(randomy[i]<np.max(ys))):
                    costs[i]+=np.sqrt((self.x-randomx[i])**2+(self.y-randomy[i])**2)*1
                    angle=(-a[i]+self.ltheta-self.theta)*180/np.pi
                    costs[i]+=np.abs((np.mod(angle+180,360)-180))*1
                    hits= beam_map2(randomx[i],randomy[i],randoma[i])
                    costs[i]+=np.sum(self.explore_map[hits == HIT_VIEW])*1
                    if(self.explore_map[gu(randomy[i]+self.y0),gu(randomx[i]+self.x0)]>0.2):
                        costs[i]+=1000
                    if(self.occ_map[gu(randomy[i]+self.y0),gu(randomx[i]+self.x0)]>0.2):
                        costs[i]+=10000
                                  
        
        #weights = 1.0/costs
        #ps = weights/weights.sum()
        #j = np.random.choice(len(costs), p=ps)
        j = np.argmin(costs)

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.orientation.w = 1.0
        msg.pose.position.x = randomx[j]
        msg.pose.position.y = randomy[j]
        msg.pose.orientation.z =a[j]
        #pub.publish(msg)
        self.pub_goal.publish(msg)
        print(randomx[j])
        print(randomy[j])
        print(a[j]*180/np.pi)
        print(costs[j])
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = randomx[j]
        marker.pose.position.y = randomy[j]
        self.pub_point.publish(marker)


        markerArray = MarkerArray()
        for i in range(len(randomx)):
            markera = Marker()
            markera.header.frame_id = "/map"
            markera.type = marker.SPHERE
            markera.action = marker.ADD
            markera.scale.x = 0.05
            markera.scale.y = 0.05
            markera.scale.z = 0.05
            markera.color.a = 1.0
            markera.color.r = 0.0
            markera.color.g = 1.0
            markera.color.b = 0.0
            markera.pose.orientation.w = 1.0        
            markera.id=i
            markera.pose.position.x = randomx[i]
            markera.pose.position.y = randomy[i]
            markerArray.markers.append(markera)

        self.pub_allpoints.publish(markerArray)


       

    def publish_map(self):
        (h,w) = self.explore_map.shape
        map_msg = OccupancyGrid()
        map_msg.info.resolution = UNIT
        map_msg.info.width = w
        map_msg.info.height = h
        map_msg.info.origin.position.x = -self.x0
        map_msg.info.origin.position.y = -self.y0
        map_msg.data = (1 + 97*self.explore_map.reshape((h*w,))).astype(np.int8)        
        self.pub_map.publish(map_msg)


class OccupancyGrid2(np.ndarray):

    def _line_idx(self, (p0x,p0y,p1x,p1y)):
        h, w = self.shape
        xs, ys = np.meshgrid(si(np.arange(w)) - p0x,
                             si(np.arange(h)) - p0y)
        vx, vy = p0x-p1x, p0y-p1y
        vlen2 = vx**2 + vy**2
        nx, ny = vy, -vx
        s = (nx*xs + ny*ys) / np.sqrt(vlen2)
        t = -(vx*xs + vy*ys)
        return (np.abs(s) < 0.03)*(0 < t)*(t < vlen2)


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


def loadtxt(f, cls=OccupancyGrid2, dtype=np.uint8):
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
    
    return grid, x0, y0,xmin,xmax,ymin,ymax

def main(args=sys.argv[1:]):
    rospy.init_node('covermapper')
    logcolor.basic_config(level=logging.INFO)
    args = rospy.myargv(args)

    if not use_jit:
        log.warn('numba installed or failed to import. '
                 'mapping will be vely slow!')

    dfl_map_fn = path.expanduser('~/map.txt')
    mb = CoverageMapBuilder((args[:1] + [dfl_map_fn])[0])


    rate = rospy.Rate(4)
    while not rospy.is_shutdown():
            mb.publish_map()
            rate.sleep()


if __name__ == "__main__":
    main()
