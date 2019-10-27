from __future__ import print_function, unicode_literals, division

import logging
from contextlib import contextmanager
from timeit import default_timer

log = logging.getLogger(__package__)

@contextmanager
def timed_code(name=None, print=print):
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
