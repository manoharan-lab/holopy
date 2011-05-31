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
'''
Fit a hologram using a genetic algorithm

..moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
..moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
'''

from numpy.random import normal, rand, permutation, randint
import numpy as np

def _minimize(target, forward_holo, parameters, generations=5, members = 10,
              mutation_prob = .2, mutation_factor = 1e-1, crossover_prob = .2):
    '''
    Minimize a set of parameters given a residual function using a genetic algorithm
    
    Parameters
    ----------
    generations : int
       number of iterations before giving up
    members : int
       number of members in the population
    '''
    E = np.zeros([generations,members]);
    averageE = np.zeros([generations]);
    bestE = np.zeros([generations]);
    F = np.zeros([generations,members]);
    averageF = np.zeros([generations]);
    bestF = np.zeros([generations]);

    def cost_modified(values):
        guess = forward_holo(values)
        return abs(abs(guess-1)-abs(target-1)).sum()/guess.size

    def cost_lsq(values):
        guess = forward_holo(values)
        return (guess*guess).sum()

    cost = cost_modified
    
    # history holds the parameter values across the evolution
    # history[generation, individual, parameter]
    history = np.zeros([generations,members,len(parameters)])

    ## Set the initial population
    # We are given an initial guess, so make the population near that guess.  We
    # will have different individuals differing distances from the guess because
    # we don't know how good the guess is
    for individual in range(members):
        # have some guesses fairly close to the initial guess, some further away
        sigma_factor = 1e-4 * 10.0 ** (individual/10.0)
        for par in range(len(parameters)):
            value = parameters[par].value
            history[0, individual, par] = normal(value, value*sigma_factor)

        
        E[0,individual] = cost(history[0,individual,:])
        F[0,individual] = 1./E[0,individual] ** 10
    averageE[0] = E[0,:].mean()
    averageF[0] = F[0,:].mean()

    ## Now for some evolution
    for gen in range(1,generations):
        print("Generation {0}".format(gen))
        ############################################
        #reproduction
        ############################################   
        scaled_F = F[gen-1,:]/sum(F[gen-1,:])
        dividers = scaled_F.cumsum()

        # TODO switch this to generating members rand's at once, do vectorized update
        for n in range(members):
            U = rand()
            replacement = 0
            while U > dividers[replacement]:
                replacement += 1

            history[gen, ...] = history[gen-1, replacement, :]

        for individual in range(members):
            ##############################################
            #mutation
            ##############################################
            if rand() < mutation_prob:
                for par in range(len(parameters)):
                    value = history[gen, individual, par]
                    history[gen, individual, par] = normal(value, value *
                                                           mutation_factor)

            ##############################################
            #crossover
            ############################################## 
#            if rand() < crossover_prob:
#                # pick some random subset of this individual's parameters to
#                # cross over
#                crossover_pars = permutation(len(parameters))[0:randint(len(parameters))]#
#                target = randint(len(parameters))
#
#                temp = history[gen, target,:].take(crossover_pars)
#                history[gen, target,:].put(history[gen, individual,
#                                                   :]take(crossover_pars),
#                                           crossover_pars)
#
#                history[gen, individual, :].put(temp, crossover_pars)


            E[gen,individual] = cost(history[gen,individual,:])
            F[gen,individual] = 1./E[gen,individual] ** 10

        averageE[gen] = E[gen,:].mean()
        averageF[gen] = F[gen,:].mean()
        bestE[gen] = E[gen,:].min()
        bestF[gen] = F[gen,:].max()

        print("Energy: Best: {0}, Average: {1}".format(bestE[gen], averageE[gen]))
        print("Fitness: Best: {0}, Average: {1}".format(bestF[gen],
                                                        averageF[gen]))
        best_individual = F[gen,:].argmax()
        best_individual = history[gen, best_individual, :]
        print("Best individual: {0}".format(best_individual))

    best = F.argmax()
    best_gen, best_individual = best // members, best % members
    single_best = history[best_gen, best_individual, :]
    single_best_error = E[best_gen, best_individual]
    print('single best: {0}'.format(single_best))
    print('best error: {0}'.format(single_best_error))

    for i in range(len(parameters)):
        parameters[i].fit_value = single_best[i]
        parameters[0].fit_error = -1 # we don't really define individual errors
# for parameters in this fit method, so just put an invalid number for now -tgd 2011-05-18
        
    return GeneticResult(history, generations, members, mutation_prob,
                         mutation_factor, crossover_prob, single_best, single_best_error)
        
class GeneticResult(object):
    def __init__(self, history, generations, members, mutation_prob, mutation_factor,
                 crossover_prob, result, fit_error):
        self.history = history
        self.generations = generations
        self.members = members
        self.mutation_prob = mutation_prob
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.fit_status = None
        self.fit_error = fit_error
        self.result = result

    def fitter_state_dict(self):
        return self.__dict__
