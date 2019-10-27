import sys
import logging
import numpy as np
from os import path
from itertools import chain

import rospy
from . import grids


np.set_printoptions(linewidth=100)
log = logging.getLogger(__package__)


def main(args=sys.argv[1:]):
    dfl_map_fn = path.expanduser('~/map.txt')
    args = ['ros'] + rospy.myargv(args)
    grid = grids.loadtxt((args + [dfl_map_fn])[2])
    #print(grid.view(np.ndarray))
    #imsave('g.png', grid)
    #m = MockRobot()
    #m.loop(m.move_path([(1, 0), (0, 0)]))
    #goal = 1.17556917667, 1.5309920311
    #home = 0.749456763268, 2.27211785316
    #home = 0.214779824018, 0.199462264776
    start_point = 0.226310253143, 0.227144956589   
    inside_spiral = 1.16639280319, 1.53181743622 
    home = start_point #inside_spiral
    mod, attr = 'mrbrain.{}.Robot'.format(args[0]).rsplit('.', 1)
    cls = getattr(__import__(mod, fromlist=[attr]), attr)
    m = cls(grid=grid, home=home)
    it = m.sleep(0.5)
    #it = chain(it, m.go_home())
    if args[1] == 'retrieve':
        it = chain(it, m.retrieve_all())
        #it = m.retreating(it, timeout=5*60.0)
    elif args[1] == 'explore':
        #it = chain(it, m.move_planned((2.12736129761, 4.49547052383)))
        it = chain(it, m.explore())
    else:
        raise ValueError(args[1])
    #it = m.move_goals()
    #it = chain(it, m.move_planned(goal))
    #it = chain(it, m.go_home())
    #it = m.move_path(cycle(np.array([(0.1,0.1), (0.2,0.0), (0.1,-0.1), (0.0,0.0)]) + start))
    #it = m.move_path(cycle((goal, start)))
    #it = m.retreating(it, timeout=3*60.0)
    #it = m.retreating(it, timeout=5*60.0)
    #it = m.visualizing(it)
    m.loop(it)


if __name__ == "__main__":
    main()
