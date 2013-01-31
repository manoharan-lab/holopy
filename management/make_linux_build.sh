#!/bin/sh

rm -r ../build
pushd .
cd ../docs
make clean
make html
cd ..
python setup.py build
python setup.py build_ext --inplace
tar cvf holopy-2.0.0.linux-x86_64.tar.gz COPYING AUTHORS build/* docs/source/* docs/build/* holopy/* management/* run_nose.py setup.py
