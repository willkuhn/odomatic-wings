# -*- coding: utf-8 -*-
"""
===================================================
Automatic identification framework (:mod: `autoID`)
===================================================

..currentmodule:: autoID

This package contains the functions needed to build and use an automatic
identification system.

Image preprocessing :mod:`autoID.preprocessing`
===============================================

.. module:: autoID.preprocessing

.. autosummary::
   :toctree: generated/

   get_image_mask - Mask an image
   sort_masks - Sort
   apply_mask -
   bounding_box -
   image_crop -
   make_horizontal -
   resize -
   pad_and_assemble -

Feature extraction :mod:`autoID.extraction`
===========================================

.. module:: autoID.extraction

.. autosummary::
   :toctree: generated/

   color2chrom -
   chrom_sample -
   gabor_sample -
   masks_from_square -
   wingAreaRatio -
   wingElongation -
   antePostRatio -
   proxDistRatio -
   regressionOfThickness -
   widestColumn -
   parabolaParams -

Data processing and model building :mod:`autoID.modelBuilding`
===========================================

.. module:: autoID.modelBuilding

.. autosummary::
    :tocree: generated/

    _str2num - extract a number from a string
    truncate_train_features -
    truncate_test_features -

Utility functions :mod:`autoID.utils`
===========================================

.. module:: autoID.utils

.. autosummary::
    :tocree: generated/

    meanImage -
"""

# Copyright (C) 2015 William R. Kuhn
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from . preprocessing import *
from . extraction import *
from . modelBuilding import *
from . utils import *
from . classifier import *


__version__ = '0.1.0'
__author__ = "William R Kuhn, willkuhn@crossveins.com"
__all__ = [s for s in dir()]
#__all__ = [s for s in dir() if not s.startswith('_')]
