#! /usr/bin/env python
from subprocess import call
import sys
import multiprocessing

t = ['nosetests', '-a', '!slow']

if len(sys.argv) > 1 and sys.argv[1] == 'coverage':
    t.extend(['--with-coverage', '--cover-package=holopy'])
else:
    t.extend(['--processes={0}'.format(multiprocessing.cpu_count())] +
             sys.argv[2:])

print(' '.join(t))
call(t)
