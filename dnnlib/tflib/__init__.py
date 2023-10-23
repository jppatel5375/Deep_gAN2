# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation,
# and any modifications thereto. Any use, reproduction, disclosure, or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Import necessary modules for TensorFlow 2.x
from . import autosummary
from . import custom_ops
from . import network
from . import optimizer
from . import tfutil

# Import specific functions and classes
from .tfutil import convert_images_to_uint8
from .tfutil import get_plugin
from .network import Network
from .optimizer import Optimizer

# If you need any specific TensorFlow 2.x functions or classes, import them here

# Make sure to update the code to work with TensorFlow 2.x conventions
# ...

# End of the updated code
