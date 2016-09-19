#!/usr/bin/python
# -*- coding: utf-8 -*-

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
'''
Graphical user interface to explore how single sphere holograms
change with the various parameters that they depend on.

This can be run from a terminal as:
python gui_manipulate.py

Can also be run from within an ipython session as:
run gui_manipulate

Takes about 5 seconds to open.

.. moduleauthor:: Rebecca W. Perry <perry.becca@gmail.com>
'''
import sys
from PyQt4 import QtGui, QtCore
from PIL.ImageQt import ImageQt

import holopy as hp
from holopy.scattering.scatterer import Sphere
from holopy.core import ImageSchema, Optics
from holopy.scattering.theory import Mie
from holopy.propagation import propagate

import scipy
import numpy as np
import Image
import time

#TODO: tie schema back to limits of z box, slider moves, lcd value stays the same
#TODO: tie schema back to reasonable radii, slider moves, lcd value stays the same

#medium Index and wavelength boxes still aren't tied in right
#careful of having large radius when wavelength gets small and makes current radius 
#outside radius range


class SingleHolo(QtGui.QWidget):
    '''
    Display single sphere hologram with interactive capability
    to modify parameters used to calculate the hologram.
    '''
    
    def __init__(self):
        super(SingleHolo, self).__init__()
        self.initUI()

    def initUI(self):

        #main image
        self.label = QtGui.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        #to display syntax of sphere and schema
        spheretitle = QtGui.QLabel()
        spheretitle.setText('Holopy Scatterer Syntax:')
        spheretitle.setStyleSheet('font-weight:bold')
        spheretitle.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
        self.sphObject = QtGui.QLabel(self)
        self.sphObject.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
        self.sphObject.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.schemaObject = QtGui.QLabel(self)
        self.schemaObject.setWordWrap(True)
        self.schemaObject.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        #warning to be used when hologram needs to be recalculated
        self.warning = QtGui.QLabel(self)
        self.warning.setText('')
        self.warning.setStyleSheet('font-size: 20pt; color: red')
        self.warning.setGeometry(30,300,350,100)
        self.warning.setWordWrap(True)

        #schema adjustment controls
        schemacontrol = QtGui.QHBoxLayout()
        wave = QtGui.QLabel(self)
        wave.setText('Wavelength:')
        self.waveText = QtGui.QLineEdit(self)
        self.waveText.setText('.660')
        self.waveText.textChanged.connect(self.warningChange)
        self.waveText.textChanged.connect(self.lengthscaleChange)

        mindex = QtGui.QLabel(self)
        mindex.setText('Medium  Index:')
        self.mindexText = QtGui.QLineEdit(self)
        self.mindexText.setText('1.33')
        self.mindexText.textChanged.connect(self.warningChange)
        self.mindexText.textChanged.connect(self.lengthscaleChange)

        pxsize = QtGui.QLabel(self)
        pxsize.setText('Pixel Spacing:')
        self.pxsizeText = QtGui.QLineEdit(self)
        self.pxsizeText.setText('0.1')
        self.pxsizeText.textChanged.connect(self.warningChange)
        self.pxsizeText.textChanged.connect(self.rangeChange)
        self.scale = float(self.pxsizeText.text())

        #sphere parameter adjustment controls
        x = QtGui.QLabel(self)
        x.setText('x')
        x.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        self.lcd = QtGui.QLCDNumber(self)
        self.lcd.setGeometry(470,10,100,30)
        self.lcd.setMinimumSize(1,30)
        start = 100
        self.lcd.display(start*self.scale)
        self.sld = QtGui.QSlider(QtCore.Qt.Horizontal,self)
        self.sld.setGeometry(470,40,100,30)
        self.sld.setMinimum(0)
        self.sld.setMaximum(256)
        self.sld.setSliderPosition(start)
        self.sld.valueChanged.connect(self.pixelScaling)
        self.sld.valueChanged.connect(self.slide)


        y = QtGui.QLabel(self)
        y.setText('y')
        y.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        self.lcd2 = QtGui.QLCDNumber(self)
        self.lcd2.setGeometry(470, 80,100,30)
        self.lcd2.setMinimumSize(1,30)
        self.lcd2.display(start*self.scale)
        self.sld2 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld2.setGeometry(470,110,100,30)
        self.sld2.setMinimum(0)
        self.sld2.setMaximum(256)
        self.sld2.setSliderPosition(start)
        self.sld2.valueChanged.connect(self.pixelScaling)
        self.sld2.valueChanged.connect(self.slide)


        z = QtGui.QLabel(self)
        z.setText('z')
        z.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        start = 30
        self.lcd3 = QtGui.QLCDNumber(self)
        self.lcd3.setGeometry(470,150,100,30)
        self.lcd3.display(30)
        self.lcd3.setMinimumSize(1,30)
        self.sld3 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld3.setGeometry(470,180,100,30)
        self.sld3.setMinimum(0)
        self.sld3.setMaximum(100)
        self.sld3.setSliderPosition((start-5)*2)
        self.sld3.valueChanged.connect(self.pixelScaling)
        self.sld3.valueChanged.connect(self.slideZ)


        radius = QtGui.QLabel(self)
        radius.setText('radius')
        radius.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        start = .5
        self.lcd4 = QtGui.QLCDNumber(self)
        self.lcd4.setGeometry(470, 220,100,30)
        self.lcd4.display(start)
        self.lcd4.setMinimumSize(1,30)
        self.sld4 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld4.setGeometry(470,250,100,30)
        self.sld4.setMinimum(0)
        self.sld4.setMaximum(100)
################### #TODO modify this part to make sliders robust
        self.sld4.setSliderPosition((start-.25)*100)
        self.sld4.valueChanged.connect(self.pixelScaling)
        self.sld4.valueChanged.connect(self.warningChange)
###################

        index = QtGui.QLabel(self)
        index.setText('sphere index')
        index.setWordWrap(True)
        index.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        start = 1.6
        self.lcd5 = QtGui.QLCDNumber(self)
        self.lcd5.setGeometry(470,290,100,30)
        self.lcd5.display(1.6)
        self.lcd5.setMinimumSize(1,30)
        self.sld5 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld5.setGeometry(470,320,100,30)
        self.sld5.setMinimum(0)
        self.sld5.setMaximum(30)
        self.sld5.setSliderPosition((start-1.0)*20)
        self.sld5.valueChanged.connect(self.pixelScaling)
        self.sld5.valueChanged.connect(self.warningChange)

        self.compute = QtGui.QPushButton('Calculate', self)
        self.compute.setDefault(True)
        self.compute.setFixedHeight(40) #attribute from qwidget class
        self.compute.clicked.connect(self.calculateHologram)

        self.timer = QtGui.QLabel(self)
        self.timer.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        #calculate starting image
        self.calculateHologram()

        #################
        #LAYOUT
        #################

        #xcontroller
        hbox0 = QtGui.QHBoxLayout()
        hbox0.addStretch(1)
        hbox0.addWidget(x)
        vbox0 = QtGui.QVBoxLayout()
        vbox0.addWidget(self.lcd)
        vbox0.addWidget(self.sld)
        hbox0.addLayout(vbox0)

        #ycontroller
        hbox1 = QtGui.QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(y)
        vbox1 = QtGui.QVBoxLayout()
        vbox1.addWidget(self.lcd2)
        vbox1.addWidget(self.sld2)
        hbox1.addLayout(vbox1)

        #zcontrol
        hbox2 = QtGui.QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(z)
        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self.lcd3)
        vbox2.addWidget(self.sld3)
        hbox2.addLayout(vbox2)

        #radius control
        hbox3 = QtGui.QHBoxLayout()
        hbox3.addStretch(1)
        hbox3.addWidget(radius)
        vbox3 = QtGui.QVBoxLayout()
        vbox3.addWidget(self.lcd4)
        vbox3.addWidget(self.sld4)
        hbox3.addLayout(vbox3)

        #index control
        hbox4 = QtGui.QHBoxLayout()
        hbox4.addStretch(1)
        hbox4.addWidget(index)
        vbox4 = QtGui.QVBoxLayout()
        vbox4.addWidget(self.lcd5)
        vbox4.addWidget(self.sld5)
        hbox4.addLayout(vbox4)

        #calculate button
        hbox5 = QtGui.QHBoxLayout()
        hbox5.addStretch(1)
        hbox5.addWidget(self.compute)

        #timer
        hbox6 = QtGui.QHBoxLayout()
        hbox6.addStretch(1)
        hbox6.addWidget(self.timer)

        #puts all the parameter adjustment buttons together
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox0)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        vbox.addLayout(hbox6)
        vbox.addStretch(1)

        contentbox = QtGui.QHBoxLayout()
        contentbox.addWidget(self.label) #hologram image

        contentbox.addLayout(vbox)

        textbox = QtGui.QVBoxLayout()
        textbox.addWidget(spheretitle)
        textbox.addWidget(self.sphObject)
        textbox.addStretch(1)
        textbox.addWidget(self.schemaObject)

        schemacontrol.addWidget(wave)
        schemacontrol.addWidget(self.waveText)
        schemacontrol.addWidget(mindex)
        schemacontrol.addWidget(self.mindexText)
        schemacontrol.addWidget(pxsize)
        schemacontrol.addWidget(self.pxsizeText)
        schemacontrol.addStretch(1)

        largevbox = QtGui.QVBoxLayout()
        largevbox.addLayout(contentbox)
        largevbox.addLayout(schemacontrol)
        largevbox.addStretch(1)
        largevbox.addLayout(textbox)
        
        self.setLayout(largevbox)
        
        self.setGeometry(300, 300, 600, 600) #window size and location
        self.setWindowTitle('Interactive Hologram')    
        self.show()

    def warningChange(self):
        #any time parameters were changed without live update
        self.warning.setText('Press calculate once you have all your parameters set')


    def lengthscaleChange(self):

        inmedwave = float(self.waveText.text())/float(self.mindexText.text())
        #radius
        radiuslow = 0.25*inmedwave
        radiushigh = 3.0*inmedwave
        #get current radius
        radius = self.lcd4.value()
        print(radius)
        #reposition slider to maintain starting value

        self.sld4.setSliderPosition((radius-radiuslow)/(radiushigh-radiuslow)*100)


    def slide(self, value):
        '''
        When x or y are changed, instead of recomputing the hologram, we
        use the shortcut of selection a region of a larger pre-computed hologram.
        '''
        source = self.sender()

        #select area to display
        x = round(self.lcd.value()/self.scale)
        y = round(self.lcd2.value()/self.scale)
        im = scipy.misc.toimage(self.holo[256-x:512-x,256-y:512-y]) #PIL image

        #convert image to pixmap
        #https://github.com/shuge/Enjoy-Qt-Python-Binding/blob/master/image/display_img/pil_to_qpixmap.py
        if im.mode == "RGB":
            pass
        elif im.mode == "L":
            im = im.convert("RGBA")
        data = im.tostring('raw',"RGBA")
        qim = QtGui.QImage(data, 256, 256, QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)

        #asign to the hologram
        myScaledPixmap = pixmap.scaled(QtCore.QSize(400,400))
        self.label.setPixmap(myScaledPixmap)

        #make a sphere object size of window displayed to
        #label at the bottom of the frame
        sphere2 = Sphere(n = self.lcd5.value()+.0001j, 
            r = self.lcd4.value(), center = (self.lcd.value(),
            self.lcd2.value(), self.lcd3.value()))
        self.sphObject.setText(repr(sphere2))

    def slideZ(self, value): 
        #using reconstructions-- better to use electric field?
        '''
        When z is changed, instead of recomputing the hologram, we
        use the shortcut of reconstructing the last computed hologram.
        '''
        source = self.sender()

        start = time.time()

        if self.lastZ == self.lcd3.value():
            self.holo = self.lastholo

        if self.lastZ < self.lcd3.value():
            self.holo = self.lastholo
            self.warning.setText('Press calculate once you have all your parameters set')
        
        if self.lastZ > self.lcd3.value(): #reconstructing a plane between hologram and object
            self.holo = np.abs(propagate(self.lastholo, -self.lcd3.value()+self.lastZ))
            self.warning.setText('Reconstruction: hologram is approximate')

        end = time.time()


        #now take only the part that you want to display
        x = round(self.sld.value())
        y = round(self.sld2.value())
        selection = self.holo[256-x:512-x,256-y:512-y]
        im = scipy.misc.toimage(selection) #PIL image

        #https://github.com/shuge/Enjoy-Qt-Python-Binding/blob/master/image/display_img/pil_to_qpixmap.py
        #myQtImage = ImageQt(im)
        #qimage = QtGui.QImage(myQtImage)
        if im.mode == "RGB":
            pass
        elif im.mode == "L":
            im = im.convert("RGBA")
        data = im.tostring('raw',"RGBA")
        qim = QtGui.QImage(data, 256, 256, QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)

        myScaledPixmap = pixmap.scaled(QtCore.QSize(400,400))

        self.label.setPixmap(myScaledPixmap)
	    #self.label.setGeometry(10, 10, 400, 400)
        #self.label.setScaledContents(True)

        self.timer.setGeometry(420,400,150,30)
        self.timer.setText('Calc. Time: '+str(round(end-start,4))+' s')
        self.timer.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        sphere2 = Sphere(n = self.lcd5.value()+.0001j, 
            r = self.lcd4.value(), center = (self.lcd.value(),
            self.lcd2.value(),self.lcd3.value()))

        self.sphObject.setText(repr(sphere2))


    def pixelScaling(self, value): #modify integer value scroll bars to meaningful units for user
        scale = self.scale
        sender = self.sender()

        if sender == self.sld:
            self.lcd.display(round(value*scale,2))

        if sender == self.sld2:
            self.lcd2.display(round(value*scale,2))

        if sender == self.sld3:
            self.lcd3.display(round(value/2.0+5,2)) #z

        if sender == self.sld4:
            inmedwave = float(self.waveText.text())/float(self.mindexText.text())
            radiuslow = .25*inmedwave
            radiushigh = 3*inmedwave
            self.lcd4.display(round((value/100.0)*(radiushigh-radiuslow)+radiuslow,2)) #r

        if sender == self.sld5:
            self.lcd5.display(round(value/20.0+1.0,2)) #index


    def rangeChange(self, value):
        self.scale = float(value)
        self.lcd.display(round(self.sld.value()*self.scale,2))
        self.lcd2.display(round(self.sld2.value()*self.scale,2))


    def calculateHologram(self): #calculate hologram with current settings
        self.warning.setText('Calculating...')

        #self.compute.setChecked(True)
        scale = self.scale
        sender = self.sender()

        ######## hologram calculation (4x big to allow scrolling)
        start = time.time()
        sphere = Sphere(n = self.lcd5.value()+.0001j, 
            r = self.lcd4.value(), 
            center = (256*self.scale,256*self.scale,self.lcd3.value()))

        sphere2 = Sphere(n = self.lcd5.value()+.0001j, 
            r = self.lcd4.value(), 
            center = (self.lcd.value(),self.lcd2.value(),self.lcd3.value()))

        self.sphObject.setText(repr(sphere2))

        schema = ImageSchema(shape = 512, spacing = float(self.pxsizeText.text()),
		    optics = Optics(wavelen = float(self.waveText.text()), 
            index = float(self.mindexText.text()), polarization = [1,0]))

        schema2 = ImageSchema(shape = 256, spacing = float(self.pxsizeText.text()),
            optics = Optics(wavelen = float(self.waveText.text()), 
            index = float(self.mindexText.text()), polarization = [1,0]))

        self.schemaObject.setText(str(repr(schema2)))
        self.holo = Mie.calc_holo(sphere, schema)
        self.lastholo = self.holo
        self.lastZ = self.lcd3.value()

        end = time.time()

        #now take only the part that you want to display
        x = round(self.sld.value())
        y = round(self.sld2.value())
        im = scipy.misc.toimage(self.holo[256-x:512-x,256-y:512-y]) #PIL image

        #https://github.com/shuge/Enjoy-Qt-Python-Binding/blob/master/image/display_img/pil_to_qpixmap.py
        #myQtImage = ImageQt(im)
        #qimage = QtGui.QImage(myQtImage)
        if im.mode == "RGB":
            pass
        elif im.mode == "L":
            im = im.convert("RGBA")
        data = im.tostring('raw',"RGBA")
        qim = QtGui.QImage(data, 256, 256, QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)

        myScaledPixmap = pixmap.scaled(QtCore.QSize(400,400))

        self.warning.setText('')
        self.label.setPixmap(myScaledPixmap)

        #self.label.setScaledContents(True)
        #myScaledPixmap.scaledToWidth(True)

        self.timer.setText('Calc. Time: '+str(round(end-start,4))+' s')
        self.timer.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = SingleHolo()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
