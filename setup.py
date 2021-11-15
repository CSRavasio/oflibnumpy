#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: 2021, Claudio S. Ravasio
# License: MIT (https://opensource.org/licenses/MIT)
# Author: Claudio S. Ravasio, PhD student at University College London (UCL), research assistant at King's College
# London (KCL), supervised by:
#   Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
#       Imaging Sciences (BMEIS) at King's College London (KCL)
#   Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK
#
# This file is part of oflibnumpy

from setuptools import setup


with open("README.rst", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='oflibnumpy',
    version='1.1.0',
    description='Optical flow library using a custom flow class based on NumPy arrays',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/RViMLab/oflibnumpy',
    author='Claudio S. Ravasio',
    author_email='claudio.s.ravasio@gmail.com',
    license='MIT',
    classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 5 - Production/Stable',

      # Pick your license as you wish (should match "license" above)
      'License :: OSI Approved :: MIT License',

      # Specify the Python versions you support here. In particular, ensure
      # that you indicate whether you support Python 2, Python 3 or both.
      'Programming Language :: Python :: 3',
    ],
    keywords='optical flow, flow, flow field, flow composition, flow combination, flow visualisation',
    project_urls={
        'Documentation': 'https://oflibnumpy.rtfd.io',
        'Source': 'https://github.com/RViMLab/oflibnumpy',
        'Tracker': 'https://github.com/RViMLab/oflibnumpy/issues',
    },
    package_dir={'': 'src'},
    packages=['oflibnumpy'],
    install_requires=[
        'numpy',
        'opencv-python',
        'scipy',
    ],
    python_requires='>=3.7',
)
