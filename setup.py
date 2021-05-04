#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: 2021, Claudio S. Ravasio
# License: GPL 3 (https://www.gnu.org/licenses/gpl-3.0.en.html)
# Author: Claudio S. Ravasio, PhD student at University College London (UCL), research assistant at King's College
# London (KCL), supervised by:
#   Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
#       Imaging Sciences (BMEIS) at King's College London (KCL)
#   Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK
#
# This file is part of oflibnumpy

from setuptools import setup


setup(
      name='oflibnumpy',
      version='1.0',
      description='Optical flow library using a custom flow class based on NumPy arrays',
      author='Claudio Ravasio',
      author_email='claudio.s.ravasio@gmail.com',
      packages=['oflibnumpy'],
      install_requires=[
            'numpy',
            'opencv-python',
            'scipy',
            'typing~=3.7.4.3',
            'sphinx~=3.5.2'
      ]
)
