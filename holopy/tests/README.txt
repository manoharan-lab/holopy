To run these tests, open a terminal, navigate to the directory containing these tests, and type: 

nosetests -a '!slow' 
or
nosetests
or
nosetests -a fast

the '!slow' version takes about 1 min, the full version takes about 5 min, and the fast version takes < 4s.

If you haven't installed holopy but have compiled the extensions using
"python setup.py build_ext --inplace", run the nosetests commands from
the directory above the holopy directory.  

To test how much of the codebase is covered by the tests, install the
python "coverage" module (python-coverage on Ubuntu) and run

nosetests -a '!slow' --with-coverage --cover-package=holopy
