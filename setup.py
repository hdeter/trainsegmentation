# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:35:47 2022

@author: hdeter

# Copyright (C) 2022 Heather S. Deter <hdeter2013@gmail.com>
# License: 3-clause BSD
"""


from setuptools import setup, find_packages

VERSION = '0.0.9' 
DESCRIPTION = 'Train Image Segmentation'
LONG_DESCRIPTION = 'Functions written to streamline generating image feature sets and training sci-kit learn classifiers'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="trainsegmentation", 
        version=VERSION,
        author="Heather S. Deter",
        author_email="<hdeter2013@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy','scipy','scikit-image','scikit-learn'], 
        keywords=['python', 'sci-kit learn','image segmentation','image classification'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering"
        ]
)