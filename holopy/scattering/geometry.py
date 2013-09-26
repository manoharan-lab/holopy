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
Routines for common calculations and transformations of groups of spheres.

This code is in need of significant refactoring and simplification, refactoring
which may break code that depends on it.

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>

"""
from __future__ import division

import numpy as np
from numpy import sqrt
from holopy.core.math import cartesian_distance
from .scatterer import Sphere, Spheres

def distances(cluster, gaponly=False):
    """
    calculate the distances between each sphere in a cluster and each of the others

    Parameters
    ----------
    cluster: :class:`holopy.scattering.scatterer.Scatterer`
        A sphere cluster to determine the interparticle distances of.
    gaponly: bool
        Whether to calculate the distances between particle centers
        or between particle surfaces (gap distances).

    Notes
    -----
    The returned array of distances includes redundant information.
    The identical distances between sphere 1 and sphere 2 and between sphere 2
    and sphere 1 are both in the returned array. Calculating and returning
    the full array makes it easy for the user to access all the interparticle
    distances starting from any sphere of interest.

    """
    num = len(cluster.centers)
    dist = np.zeros([num,num])
    for i in np.arange(0,num):
        for j in np.arange(0,num):
            dist[i,j] = cartesian_distance(cluster.centers[i,:],cluster.centers[j,:])
            #modification to change center to center distances
            #to gap distances if asked for
            if gaponly==True and i!=j:
                dist[i,j] = dist[i,j]-cluster.r[i]-cluster.r[j]
    return dist

def angles(cluster, degrees=True):
    """
    calculate the angles between one particle and every pair of other particles

    Parameters
    ----------
    cluster: :class:`holopy.scattering.scatterer.Scatterer`
        A sphere cluster to determine the interparticle distances of.
    degrees: bool
        Whether to return angles in degrees (True) or in radians (False).

    Notes
    -----
    Angle abc is the acute angle formed by edges conecting points ab and bc.
    If a, b, and c are locations of particles (vertices),
    the returned 3D array has non-zero values for angles abc, zeros
    for angles aba, and NAN's for "angles" aab.

    """
    num = len(cluster.centers)
    ang = np.zeros([num,num,num])
    dist = distances(cluster)
    for i in np.arange(0,num):
        for j in np.arange(0,num): #this particle is the center of the angle
            for k in np.arange(0,num):
                Adjacent1=dist[i,j]
                Adjacent2=dist[j,k]
                Opposite=dist[i,k]
                #use the law of cosines to determine the angles from the distances
                ang[i,j,k] = np.arccos((Adjacent1**2+Adjacent2**2-Opposite**2)/(2*Adjacent1*Adjacent2))
    if degrees==True:
        ang=ang/np.pi*180.0
    return ang #ang[a,b,c] is the acute angle abc as used in geometry (be in the middle)

def make_tricluster(index,radius,gap,xcom=0,ycom=0,zcom=0):
    """
    Returns a sphere cluster of three particles forming an equilateral triangle
    centered on a given center of mass.

    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
    xcom:
        Center of mass x-coordinate
    ycom:
        Center of mass y-coordinate
    zcom:
        Center of mass z-coordinate

    """
    #currently restricted to all identical spheres
    xs = np.array([1/sqrt(3)*(radius+gap/2.0),
        1/sqrt(3)*(radius+gap/2.0),-2/sqrt(3)*(radius+gap/2.0)])+xcom
    ys = np.array([-radius-gap/2.0,radius+gap/2.0,0])+ycom
    zs = np.array(np.zeros(3))+zcom
    triangle = Spheres([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2]))])
    return triangle

def make_sqcluster(index,radius,gap,xcom=0,ycom=0,zcom=0):
    """
    Returns a sphere cluster of four particles forming a
    square centered on a given center of mass.

    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
    xcom:
        Center of mass x-coordinate
    ycom:
        Center of mass y-coordinate
    zcom:
        Center of mass z-coordinate
    """
    #currently restricted to all identical spheres
    xs = np.array([-radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.])+xcom
    ys = np.array([-radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0])+ycom
    zs = np.array([0,0,0,0])+zcom
    square = Spheres([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3]))])
    return square

def make_tetracluster(index,radius,gap,xcom=0,ycom=0,zcom=0):
    """
    Returns a sphere cluster of four particles forming a
    tetrahedron centered on a given center of mass.

    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
    xcom:
        Center of mass x-coordinate
    ycom:
        Center of mass y-coordinate
    zcom:
        Center of mass z-coordinate
    """
    #currently restricted to all identical spheres
    xs = np.array([1/sqrt(3)*(radius+gap/2.0),1/sqrt(3)*(radius+gap/2.0),
        -2/sqrt(3)*(radius+gap/2.0),0])+xcom
    ys = np.array([-radius-gap/2.0,radius+gap/2.0,0,0])+ycom
    zs = np.array([-(1/4.0)*sqrt(2/3.0)*(2*radius+gap),
        -(1/4.0)*sqrt(2/3.0)*(2*radius+gap),-(1/4.0)*sqrt(2/3.0)*(2*radius+gap),
        (3/4.0)*sqrt(2/3.0)*(2*radius+gap)])+zcom
    tetra = Spheres([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3]))])
    return tetra

def make_tribipyrcluster(index,radius,gap,xcom=0,ycom=0,zcom=0):
    """
    Returns a sphere cluster of five particles forming a triagonal bipyramid
    centered on a given center of mass.

    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
    xcom:
        Center of mass x-coordinate
    ycom:
        Center of mass y-coordinate
    zcom:
        Center of mass z-coordinate
    """
    #currently restricted to all identical spheres
    xs = [1/sqrt(3)*(radius+gap/2.0),1/sqrt(3)*(radius+gap/2.0),-2/sqrt(3)*(radius+gap/2.0),0,0]
    ys = [-radius-gap/2.0,radius+gap/2.0,0,0,0]
    zs = [0,0,0,sqrt(2/3.0)*(2*radius+gap),-sqrt(2/3.0)*(2*radius+gap)]
    triangularbipyramid = Spheres([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3])),
        Sphere(n=index, r = radius, center = (xs[4], ys[4], zs[4]))])
    return triangularbipyramid

def make_octacluster(index,radius,gap,xcom=0,ycom=0,zcom=0):
    """
    Returns a sphere cluster of six particles forming an octahedron centered on
    a given center of mass.

    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
    xcom:
        Center of mass x-coordinate
    ycom:
        Center of mass y-coordinate
    zcom:
        Center of mass z-coordinate
    """
    #currently restricted to all identical spheres
    xs = np.array([-radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.0,0,0])+xcom
    ys = np.array([-radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0,0,0])+ycom
    zs = np.array([0,0,0,0,1/sqrt(2)*(2*radius+gap),-1/sqrt(2)*(2*radius+gap)])+zcom
    octahedron = Spheres([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3])),
        Sphere(n=index, r = radius, center = (xs[4], ys[4], zs[4])),
        Sphere(n=index, r = radius, center = (xs[5], ys[5], zs[5]))])
    return octahedron

def make_cubecluster(index,radius,gap,xcom=0,ycom=0,zcom=0):
    """
    Returns a sphere cluster of eight particles forming a cube centered on a
    given center of mass.

    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
    xcom:
        Center of mass x-coordinate
    ycom:
        Center of mass y-coordinate
    zcom:
        Center of mass z-coordinate
    """
    #currently restricted to all identical spheres
    xs = np.array([-radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.0,
            -radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.0])+xcom
    ys = np.array([-radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0,
            -radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0])+ycom
    zs = np.array([-radius-gap/2.0,-radius-gap/2.0,-radius-gap/2.0,-radius-gap/2.0,
            radius+gap/2.0,radius+gap/2.0,radius+gap/2.0,radius+gap/2.0])+zcom
    cube = Spheres([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3])),
        Sphere(n=index, r = radius, center = (xs[4], ys[4], zs[4])),
        Sphere(n=index, r = radius, center = (xs[5], ys[5], zs[5])),
        Sphere(n=index, r = radius, center = (xs[6], ys[6], zs[6])),
        Sphere(n=index, r = radius, center = (xs[7], ys[7], zs[7]))])
    return cube

def make_polytetracluster(index,radius,gap,xcom=0,ycom=0,zcom=0):
    """
    Returns a sphere cluster of six particles forming a polytetrahedron centered
    on a given center of mass of the middle tetrahedron.

    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
    xcom:
        Center of mass of the middle tetrahedron x-coordinate
    ycom:
        Center of mass of the middle tetrahedron x-coordinate
    zcom:
        Center of mass of the middle tetrahedron x-coordinate
    """
    #currently restricted to all identical spheres
    #side length= 2*sqrt(2), needs to be 2*r+gap
    #1 and 3 have exactly the same distance to all the others
    xs = np.array([-1/(sqrt(2))*(radius+gap/2.0),
        1/(sqrt(2))*(radius+gap/2.0),1/(sqrt(2))*(radius+gap/2.0),
        -1/(sqrt(2))*(radius+gap/2.0),5/3.0/sqrt(2)*(radius+gap/2.0),
        -5/3.0/sqrt(2)*(radius+gap/2.0)])+xcom
    ys = np.array([1/(sqrt(2))*(radius+gap/2.0),
        1/(sqrt(2))*(radius+gap/2.0),-1/(sqrt(2))*(radius+gap/2.0),
        -1/(sqrt(2))*(radius+gap/2.0),-5/3.0/sqrt(2)*(radius+gap/2.0),
        5/3.0/sqrt(2)*(radius+gap/2.0)])+ycom
    zs = np.array([1/(sqrt(2))*(radius+gap/2.0),
        -1/(sqrt(2))*(radius+gap/2.0),1/(sqrt(2))*(radius+gap/2.0),
        -1/(sqrt(2))*(radius+gap/2.0),-5/3.0/sqrt(2)*(radius+gap/2.0),
        -5/3.0/sqrt(2)*(radius+gap/2.0)])+zcom
    polytetra = Spheres([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3])),
        Sphere(n=index, r = radius, center = (xs[4], ys[4], zs[4])),
        Sphere(n=index, r = radius, center = (xs[5], ys[5], zs[5]))])
    return polytetra
