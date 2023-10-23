# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pickle
import re

import numpy as np
import PIL.Image

import tensorflow as tf  # Import TensorFlow 2.x

from tensorflow.python.ops import array_ops

#----------------------------------------------------------------------------

def get_output_for(layer, inputs, output_transform):
    tflib.setup_external_inputs(layer, inputs)
    layer.input_shape = [1] + layer.input_shapes[1][1:]
    layer.run(layer.input_templates[0])
    return array_ops.make_ndarray(layer.output).copy()

#----------------------------------------------------------------------------

class Layer:
    def __init__(self, component, name):
        self.name = name
        self.vars = dict()
        for name, value in component.vars.items():
            if name.startswith(self.name):
                self.vars[name] = value

    def run(self, inputs, output_transform):
        outputs = get_output_for(self, inputs, output_transform)
        return outputs

#----------------------------------------------------------------------------

class Gen:
    def __init__(self, Gs):
        self.Gs = Gs

    def run(self, z, class_idx, noise_vars, label):
        component = self.Gs.clone()
        component.components.synthesis.run(
            z,
            noise_vars,
            label
        )
        images = component.components.synthesis.run(z, class_idx=class_idx, noise_vars=noise_vars, label=label)
        return images[0]

#----------------------------------------------------------------------------

class Gs:
    def __init__(self, network_pkl):
        with open(network_pkl, 'rb') as fp:
            G, D, self.Gs = pickle.load(fp)

    def clone(self):
        return Gs(self.Gs)

    def set_vars(self, vars):
        self.Gs.run(self.Gs.input_templates[0], noise_vars=vars)
        
    def input_templates(self):
        return self.Gs.input_templates
        
    def run(self, *args, **kwargs):
        return self.Gs.components.synthesis.run(*args, **kwargs)

    def components(self):
        return self.Gs.components

#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, truncation_psi, outdir, class_idx, dlatents_npz):
    print('Loading networks from "%s"...' % network_pkl)
    Gs = Gs(network_pkl)
    os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:
        print(f'Generating images from dlatents file "{dlatents_npz}"')
        dlatents = np.load(dlatents_npz)['dlatents']
        assert dlatents.shape[1:] == (18, 512)  # [N, 18, 512]
        imgs = Gs.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        for i, img in enumerate(imgs):
            fname = f'{outdir}/dlatent{i:02d}.png'
            print(f'Saved {fname}')
            PIL.Image.fromarray(img, 'RGB').save(fname)
        return

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components().synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_templates()[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    gen = Gen(Gs)

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_templates()[0].shape.as_list()[1:])  # [minibatch, component]
        for var in noise_vars:
            tflib.set_vars({var: rnd.randn(*var.shape.as_list())})
        images = gen.run(z, class_idx, noise_vars, label)
        PIL.Image.fromarray(images, 'RGB').save(f'{outdir}/seed{seed:04d}.png')

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--seeds', type=_parse_num_range,
