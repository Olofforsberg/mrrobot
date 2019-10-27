#!/usr/bin/python

from __future__ import print_function, unicode_literals

import sys
from select import select
from os import O_NONBLOCK, read
from fcntl import F_GETFL, F_SETFL, fcntl
from termios import ECHO, ICANON, TCSADRAIN, VMIN, VTIME, tcgetattr, tcsetattr
from contextlib import contextmanager

import rospy
from geometry_msgs.msg import Twist, Vector3

@contextmanager
def drained(f):
    if hasattr(f, 'fileno'):
        f = f.fileno()
    fl = fcntl(f, F_GETFL)
    fcntl(f, F_SETFL, fl | O_NONBLOCK)
    tcold = tcgetattr(f)
    tcnew = tcgetattr(f)
    tcnew[3] = tcnew[3] & ~(ECHO | ICANON)
    tcnew[6][VMIN] = 0
    tcnew[6][VTIME] = 0
    tcsetattr(f, TCSADRAIN, tcnew)
    try:
        yield f
    finally:
        tcsetattr(f, TCSADRAIN, tcold)
        fcntl(f, F_SETFL, fl)

def read_input(f):
    esc, ansi = False, False
    if hasattr(f, 'fileno'):
        f = f.fileno()
    while True:
        if select([f], [], [], 0) == ([], [], []):
            yield
            continue
        for ch in read(f, 4):
            if   ch == ' ': yield 'stop'
            elif ch == 'q': yield 'quit'
            elif ch == 'w': yield 'frwd'
            elif ch == 's': yield 'bkwd'
            elif ch == 'd': yield 'rght'
            elif ch == 'a': yield 'left'
            elif ansi and ch == 'A': yield 'frwd'
            elif ansi and ch == 'B': yield 'bkwd'
            elif ansi and ch == 'C': yield 'rght'
            elif ansi and ch == 'D': yield 'left'
            elif esc and ch == '[': pass
            elif ch == '\x1b': pass
            else: print('invalid input', (ch, esc, ansi))
            ansi = esc and ch == '['
            esc = ch == '\x1b'

def eval_input(vw, k):
    v, w = vw
    if   k == 'stop': return 0.0, 0.0
    elif k == 'quit': rospy.signal_shutdown('user quit')
    elif k == 'frwd': return v + 0.15, w
    elif k == 'bkwd': return v - 0.15, w
    elif k == 'rght': return v, w - 0.50
    elif k == 'left': return v, w + 0.50
    elif k is None:   pass
    else:             print('unknown event', k)
    return vw

def main(f=sys.stdin):
    pub = rospy.Publisher('/motor_controller/twist', Twist, queue_size=100)
    rospy.init_node('teleop', anonymous=True)
    rate = rospy.Rate(100)
    v, w = 0.0, 0.0
    print('q to quit')
    print('<space> to halt')
    print('wasd or arrow keys to increase speed')
    print()
    with drained(f):
        inputs = read_input(f)
        while not rospy.is_shutdown():
            v, w = eval_input((v, w), next(inputs))
            pub.publish(Twist(linear=Vector3(v, 0.0, 0.0),
                              angular=Vector3(0.0, 0.0, w)))
            rate.sleep()

if __name__ == '__main__':
    main()
