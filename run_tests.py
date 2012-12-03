#! /usr/bin/env python
import subprocess
import sys
import multiprocessing

subprocess.call(['nosetests', '-a', '!slow', '--processes={0}'.format(
    multiprocessing.cpu_count())] + sys.argv[2:])
