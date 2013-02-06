#!/bin/sh
# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.

bzr branch -r tag:2.0.0 lp:holopy holopy-2.0.0

cd holopy-2.0.0/docs
make html
cd ..
python setup.py build
python setup.py build_ext --inplace
cd ..
tar zcvf holopy-2.0.0.linux-x86_64.tar.gz holopy-2.0.0/*
rm -r holopy-2.0.0
