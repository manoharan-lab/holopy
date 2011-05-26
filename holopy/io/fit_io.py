# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
from .yaml_io import load_yaml
from .image_io import load
import yaml_io
import time
import yaml
import os.path
from holopy.optics import Optics
from holopy.utility.helpers import _ensure_array, _mkdir_p, _copy_file
from holopy.utility import errors
from holopy.analyze.minimizers.nmpfit_adapter import NmpfitParam, TiedNmpfitParam
import glob
from holopy.process import zero_filter, normalize, background
from holopy.process import centerfinder
from holopy.hologram import subimage
import numpy as np


class FitInputDeck(object):
    def __init__(self, filename):
        self._yaml = load_yaml(filename)
        self.filename = filename
        self.deck_location = os.path.split(filename)[0]
        self.cluster_type = _choose_cluster_type(self)
        self.fit_params = load_yaml(self._get_filename('fit_file'))
        self.optics = Optics(**load_yaml(self._get_filename('optics_file')))
        self.results_directory = os.path.join(self.deck_location,
                                              self['results_directory'])
        self.optics.index = self._yaml['medium_index']
        # Load the background on demand since it could take a little while and
        # we want to keep this initialization fast
        self._background = None

    # These functions let other code interact with FitInputDeck as if it is a
    # dictionary: deck['name'] will index into the loaded yaml file
    def has_key(self, key):
        return self._yaml.has_key(key)
    def get(self, key):
        return self._yaml.get(key)
    def __getitem__(self, key):
        return self._yaml[key]
    # Lack of a __setitem__ is intentional, we don't other code changing the
    # input deck

    def _image_name(self, num):
        return 'image' + str(num).zfill(4)

    def _get_filename(self, name):
        return os.path.join(self.deck_location, self[name])
    @property
    def background(self):
        if self._background is None:
            if self.has_key('background_file'):
                self._background = zero_filter(load(self['background_file'],
                                                    self.optics))
            else:
                self._background = None
        return self._background
    
    def _image_file_name(self, num):
        image_file = glob.glob(os.path.join(self.deck_location,
                                            self['data_directory'],
                                            self._image_name(num) +'.*'))
        if len(image_file) == 1:
            image_file=image_file[0]
        else:
            raise errors.LoadError(image_file,
                                   "Multiple files found, failing because of ambiguity")
        return image_file
    
    def _get_fit_parameters(self):
        '''
        Make dictionary of fittable parameters based on physical values in
        the input deck.
        '''

        fit_params = self.fit_params

        # user may or may not have specified to hold parameters constant, 
        # we need an empty list if they did not.  
        if not self.has_key('hold_constant'):
            self._yaml['hold_constant'] = []
        # ditto for tied
        if not self.has_key('tied'):
            self._yaml['tied'] = {}

        def make_param(name):
            # rescale into nondimensionalized scattering units because it will
            # put the parameters into a similar order of maginitude and improve
            # fitter stability
            scaling = _param_rescaling_factor(name, self.cluster_type,
                                              self.optics)
            value = self._yaml[name] * scaling

            # if the parm is specified as x_n we may only have a values sepecified
            # for x, in this case, use the general values for x

            fit_params_name = name
            if not fit_params['bounds'].has_key(name):
                fit_params_name = _split_particle_number(name)[0]
                if not fit_params['bounds'].has_key(fit_params_name):
                    raise errors.ParameterDefinitionError('Fit params not specified'
                                                          + ' for parameter: '+name)

            def scaled_fit_parm(inner_name):
                parm = fit_params[inner_name][fit_params_name]
                if parm is not None:
                    parm = _ensure_array(parm)
                    for i in range(len(parm)):
                        if parm[i] is not None:
                            parm[i] *= scaling
                    if len(parm) == 1:
                        parm = float(parm)
                return parm

            # Here, handle tied parameters
            

            return NmpfitParam(value = value,
                                fixed = name in self['hold_constant'],
                                limits = scaled_fit_parm('bounds'),
                                step = scaled_fit_parm('step'),
                                maxstep = scaled_fit_parm('max_step'),
                                name = name)

        parameters = {}
        
        for name in self._get_full_par_ordering():
            try:
                parameters[name] = make_param(name)
            except KeyError: # if parameter value not specified in input deck
                # probably because of tying, ignore for now
                pass

        if self.has_key('tied'):
            if self['tied']:
                for par, tied_to in self['tied'].iteritems():
                    parameters[par] = TiedNmpfitParam(parameters[tied_to], par)

        return parameters

    def _get_full_par_ordering(self):
        # I don't think this is right. self isn't a parameter dictionary
        #num_particles = _get_num_particles(self,
        #                                   self.cluster_type.par_ordering[0])
        # Temporary workaround
        if self._yaml['cluster_type'] == 'mie':
            num_particles = 1
        if self._yaml['cluster_type'] == 'dimer':
            num_particles = 2
        elif self._yaml['cluster_type'] == 'trimer':
            num_particles = 3

        '''
        par_ordering = []
        for name in self.cluster_type.par_ordering: 
            if self.has_key(name):
                par_ordering.append(name)
            else: # the parameter name does not match, that is probably because it
                # is a numbered parameter, and par list only contains the base name,
                # ie x -> x_1, x_2, ...
                for i in range(num_particles):
                    newname = '{0}_{1}'.format(name, i+1)
                    par_ordering.append(newname)
                    
    
        '''
        # temporary workaround
        par_ordering = self.cluster_type.par_ordering

        return par_ordering

    def _get_image_by_number(self, number=None):
        # If image number is not specified, return the first one
        if number is None:
            number = self['image_range'][0]

        # read the optics fresh, because self.optics will have resampled pixel
        # sizes for correct parameter conversions, but we don't want that when
        # loading an image
        opt = self.optics
        if self.background is not None:
            image = background(load(self._image_file_name(number),
                                    opt), self.background, 'divide')
        else:
            image = load(self._image_file_name(number), opt)

        if self.has_key('subimage_center'):
            hologuess = subimage(image, self['subimage_center'],
                                 self.get('subimage_size'))
            #use centerfinder to adjust the subimage center if desired
            if (self.get('subimage_center_autoadjust') and
                self.get('subimage_center_autoadjust').lower() == 'true'):
                #correct subimage to put particle in the center
                a = centerfinder.center_find(hologuess)
                hologuess = subimage(image, self['subimage_center']-
                                     np.array(self.get('subimage_size'))/2.+
                                     np.array([round(a[0]),round(a[1])]),
                                     #likely to within 2 pixels of the center
                                     self.get('subimage_size'))
            image = hologuess

        if self.has_key('resample'):
            image = image.resample(self['resample'])

        # Normalize after all other transformations because things like
        # subimaging could change the mean pixel value.  
        normalize(image)
            
        return image

    def _get_extra_minimizer_params(self):
        minimizer_params = {}
        def get_tol(name):
            return self.fit_params['tols'][name]
        if self.fit_params and self.fit_params.has_key('tols'):
            minimizer_params['gtol'] = get_tol('gtol')
            minimizer_params['xtol'] = get_tol('xtol')
            minimizer_params['ftol'] = get_tol('ftol')

        if self.fit_params.has_key('max_iter'):
            minimizer_params['maxiter'] = self.fit_params['max_iter']
        return minimizer_params

    def _param_rescaling_factor(self, name):
        return _param_rescaling_factor(name, self.cluster_type, self.optics)
    
    
            
def load_FitInputDeck(name):
    if isinstance(name, FitInputDeck):
        return name
    else:
        return FitInputDeck(name)

def _setup_output_directory(deck):
    # make the output directory if needed

    _mkdir_p(os.path.join(deck.results_directory,'fits'))
    
    # Copy all of the files used to the output directory, helping associate with
    # the results with the parameters used to get them
    _copy_file(deck.filename, deck.results_directory)
    _copy_file(deck._get_filename('optics_file'), deck.results_directory)
    _copy_file(deck._get_filename('fit_file'), deck.results_directory)

    
def _get_num_particles(parm_dict, name):
    num_particles = 0
    while parm_dict.get('{0}_{1}'.format(name, num_particles+1)):
        num_particles += 1
    return num_particles

def _param_rescaling_factor(name, cluster_type, optics):
    if cluster_type._scaled_by_k(name):
        # multiply the parameters that are lengths by k to
        # scaled.  This will put parameters closer to the same
        # order of magnitude and improve the linear algebra properties 
        # of the fitter
        scaling = optics.wavevec
    elif cluster_type._scaled_by_med_index(name):
        scaling = 1.0/optics.index
    else:
        scaling = 1.0
    return scaling

def _split_particle_number(name):
    tok = name.split('_')
    try:
        number = int(tok[-1])
    except ValueError:
        # if the parameter name has no number off the end, then there is only
        # one of that parameter, return none to make that clear
        return name, None
    return '_'.join(tok[:-1]), number

def _choose_cluster_type(name):
    if isinstance(name, FitInputDeck) or isinstance(name, dict):
        name = name['cluster_type']
    if name == 'mie':
        import holopy.model.mie_fortran as cluster_type
#    if name == 'mie_c':
#        import holopy.model.mie as cluster_type
    elif name == 'dimer':
        import holopy.model.tmatrix_dimer as cluster_type
    elif name == 'trimer':
        import holopy.model.tmatrix_trimer as cluster_type
    else:
        raise NotImplementedError("Fit type {0} not yet implemented.".format(name))
    return cluster_type



def _output_frame_yaml(deck, fitresult, num):
    image_file = deck._image_file_name(num)
    opt = deck.optics
    # Output: YAML file for each frame
    out_param_dict = {}
    for p in fitresult.parlist:
        out_param_dict[p.name] = p.output_dict()

    io_dict = {'fit_image' : image_file,
               'input_deck': os.path.abspath(deck.filename),
               'hologram_shape' : list(fitresult.holo_shape)}
    io_dict['fit_image'] = image_file
    if deck.has_key('background_file'):
        io_dict['bg_file'] = deck['background_file']
    else:
        io_dict['bg_file'] = None
    if deck.has_key('subimage_center'):
        io_dict['subimage_center'] = deck['subimage_center']
        io_dict['subimage_size'] = deck['subimage_size']
    else:
        io_dict['subimage_center'] = None
        io_dict['subimage_size'] = None

    output_opt_dict = {'polarization' : opt.polarization.tolist(),
                       'pixel_scale' : opt.pixel_scale.tolist(),
                       'index' : opt.index,
                       'wavelen': opt.wavelen}

    output_dict = {'model' : deck['cluster_type'],
                   'optics' : output_opt_dict,
                   'io' : io_dict,
                   'nmpfit_data' : fitresult.fitter_state_dict(),
                   'parameters' : out_param_dict}
    out_yaml_file = open(os.path.join(deck.results_directory, 'fits', 
                                      deck._image_name(num) + '_fit.yaml'), 'w')
    output_dict = yaml_io._clean_for_yaml(output_dict)
    yaml.dump(output_dict, out_yaml_file)
    out_yaml_file.close()


class SimpleFitOutFile(object):
    def __init__(self, deck):
        self.outf = open(os.path.join(deck.results_directory,
                                      'fit_result.tsv'), 'w', 1)
        # write the header
        self.outf.write('# Fit results from: {0} @ {1}\n'.format(deck.filename,
                                                                 time.asctime()))
        self.cluster_type = _choose_cluster_type(deck)
        self.output_p_ordering = deck._get_full_par_ordering()
        self.outf.write("Image\t" + '\t'.join(self.output_p_ordering))
        self.outf.write('\tfnorm\tstatus')
        self.opt = deck.optics
        self.deck = deck
        
        if deck.has_key('frame_time'):
            self.frame_time = deck['frame_time']
        elif deck.has_key('frames_per_second'):
            self.frame_time = 1.0/deck['frames_per_second']
        else:
            self.frame_time=0

        if self.frame_time:
            self.outf.write('\ttimestamp\n')
        else:
            self.outf.write('\n')

        self.line_count = 0

    def write_data_line(self, param_dict, num, fnorm, status):
        pars_for_output = {}
        image_file = self.deck._image_file_name(num)


        for key, val in param_dict.iteritems():
            pars_for_output[key] = (val['final_value'] /
                                    _param_rescaling_factor(key,
                                                            self.cluster_type,
                                                            self.opt))

        pars_for_output = [str(pars_for_output[key]) for key in 
                           self.output_p_ordering]

        self.outf.write(image_file + '\t' + '\t'.join(pars_for_output))
        self.outf.write('\t{0}\t{1}'.format(fnorm, status))
        # writes frame time based on # of frames done
        if self.frame_time:
            self.outf.write('\t{0}\n'.format(self.line_count*self.frame_time))
        else:
            self.outf.write('\n')

        self.line_count += 1

    def close(self):
        self.outf.close()
