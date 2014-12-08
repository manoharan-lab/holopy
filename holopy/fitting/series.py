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
"""
Routine for fitting a time series of holograms to an exact solution

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>

"""
from tempfile import NamedTemporaryFile
import pandas as pd
from holopy.fitting import fit as fit_single
import os

#default preprocessing function
def div_normalize(holo, bg, df, model):
    if df is None:
        df = np.zeros_like(holo)
    if bg is not None:
        imagetofit = normalize((holo-df)/(bg-df))
    else:
        imagetofit = normalize(holo)
    return imagetofit

def scatterer_centered_subimage(size, recenter_at_edge=False):
    def preprocess(holo, bg, df, model):
        center = np.array(model.scatterer.guess.center[:2])/holo.spacing
        try:
            return normalize(subimage(holo/bg, center, size))
        except IndexError:
            if not recenter_at_edge:
                raise
            new_center = np.array(model.scatterer.guess.center[:2])
            new_center -= np.clip(new_center-np.array(size)/2, -np.inf, 0)
            new_center += np.clip(holo.shape[:2]-(new_center + np.array(size)/2), -np.inf, 0)
            return normalize(subimage(holo/bg, new_center, size))

    return preprocess

#default updating function
def update_all(model, fitted_result):
    for p in model.parameters:
        name = p.name
        p.guess = fitted_result.parameters[name]
    return model

class SeriesResult(object):
    def __init__(self, file=None):
        self.file = file
        if file is None:
            file = NamedTemporaryFile
            self.file = file.name

        self.store = pd.HDFStore(file)

    def append(self, result):
        self.store.append('fit_result', result.DataFrame_row())

    def __getitem__(self, v):
        self.store.select('fit_result')[v]

    def save(self, name):
        self.store.flush()
        if self.file != name:
            os.renames(self.file, name)


def fit(model, data, bg=None, df=None, output_file=None,
        preprocess_func=div_normalize, update_func=update_all, restart=False,
        **kwargs):
    if os.path.splitext(output_file)[1] not in ['.h5', '.hdf5']:
        output_file += '.h5'
    results = SeriesResult(output_file)
    if isinstance(bg, basestring):
        bg = load(bg, spacing=data_spacing, optics=data_optis)
    if isinstance(df, basestring):
        df = load(df, spacing=data_spacing, optics=data_optics)

    for i, frame in enumerate(data):
        imagetofit = preprocess_func(frame, bg, df, model)
        result = fit_single(model, imagetofit, **kwargs)
        results.append(fit_single(model, imagetofit, **kwargs))
        print("Fit frame {}, rsq={}".format(i, results.rsq))

    return results
