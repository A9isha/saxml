# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Convert weights from a llama/vicuna model to a pax one.

Usage:

# Get LLaMA pytorch_vars from Meta 

# Example cmd:
python3 -m convert_llama_ckpt --base llama_7b --pax pax_7b
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib

import jax
import jax.numpy as jnp
from jax.experimental import pjit
import numpy as np

from paxml import checkpoints
from paxml import train_states
from praxis import py_utils

import torch
import os
from tqdm import tqdm

num_layers = 32
num_heads = 32
dims_per_head = 128
vocab = 32000
data_parallel = 4
model_parallel = 4


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype != jnp.bfloat16 else x, t)

def convert(base_model_path, pax_model_path):
  """Convert from Llama2 to pax."""
  print(f'Loading the base model from {base_model_path}')
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob('*.pth'))
  print(f'ckpt_paths={ckpt_paths}')
  pytorch_vars = {}
  for i, ckpt_path in tqdm(enumerate(ckpt_paths)):
    print(f'Loading checkpoint {i+1} of {len(ckpt_paths)} ...')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(f'torch checkpoint loaded ')
    pytorch_vars[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
    print(f'checkpoint splitting done')
  pytorch_vars = [pytorch_vars[i] for i in sorted(list(pytorch_vars.keys()))]

  print(f'pytorch_vars = {pytorch_vars}')

  jax_weights = {
      'lm': {
          'embedding_lookup': {
              'emb_var': np.concatenate([var['tok_embeddings.weight'].float().numpy() for var in pytorch_vars], axis=1)[:vocab,:]
              },
          'softmax': {
              'logits_ffn': {
                  'linear': {
                      'w': np.concatenate([var['output.weight'].float().numpy() for var in pytorch_vars], axis=0).transpose()[:, :vocab]
                      }
                  }
              },
          'final_ln': {
              'scale': pytorch_vars[0]['norm.weight'].float().numpy()
              },
          'transformer': {}
          }
      }
  for layer_idx in tqdm(range(num_layers)):
    wq = np.concatenate([var['layers.%d.attention.wq.weight' % (layer_idx)].float().numpy() for var in pytorch_vars], axis=0).transpose()
    wk = np.concatenate([var['layers.%d.attention.wk.weight' % (layer_idx)].float().numpy() for var in pytorch_vars], axis=0).transpose()
    wv = np.concatenate([var['layers.%d.attention.wv.weight' % (layer_idx)].float().numpy() for var in pytorch_vars], axis=0).transpose()
    wc = np.stack([wq, wk, wv], axis=0)
    wc = np.reshape(wc, [3, num_heads * dims_per_head, num_heads, dims_per_head])

    w_post = np.concatenate(
        [
            var['layers.%d.attention.wo.weight' % (layer_idx)].float().numpy()
            for var in pytorch_vars
        ],
        axis=1,
    )
    w_post = np.reshape(w_post, [num_heads * dims_per_head, num_heads, dims_per_head])
    layer_weight = {
        'self_attention': {
            'combined_qkv': {
                'w': wc
                },
            'post': {
                'w': w_post
                }
            },
        'ff_layer': {
            'ffn_layer1_gate': {
                'linear': {
                    'w': np.concatenate([var['layers.%d.feed_forward.w1.weight' % (layer_idx)].float().numpy() for var in pytorch_vars], axis=0).transpose()
                    }
                },
            'ffn_layer1': {
                'linear': {
                    'w': np.concatenate([var['layers.%d.feed_forward.w3.weight' % (layer_idx)].float().numpy() for var in pytorch_vars], axis=0).transpose()
                    }
                },
            'ffn_layer2': {
                'linear': {
                    'w': np.concatenate([var['layers.%d.feed_forward.w2.weight' % (layer_idx)].float().numpy() for var in pytorch_vars], axis=1).transpose()
                    }
                },
            'layer_norm': {
                'scale': pytorch_vars[0]['layers.%d.ffn_norm.weight' % (layer_idx)].float().numpy()
                }
            },
        'layer_norm': {
            'scale': pytorch_vars[0]['layers.%d.attention_norm.weight' % (layer_idx)].float().numpy()
            }
        }
    jax_weights['lm']['transformer']['x_layers_%d' % layer_idx] = layer_weight

  jax_weights = to_bf16(jax_weights)

  print(f'Saving the pax model to {pax_model_path}')
  jax_states = train_states.TrainState(
      step=np.zeros(1),
      mdl_vars={'params': jax_weights},
      opt_states={})

  device_mesh = py_utils.create_device_mesh([1, data_parallel, model_parallel])
  global_mesh = jax.sharding.Mesh(
      device_mesh, ['replica', 'data_mdl2', 'mdl'])

  # Identity pjit is needed to output a GDA model_states.
  def identity(x):
    return x

  pjitted_identity = pjit.pjit(identity,
                               in_shardings=None,
                               out_shardings=None)

  with global_mesh:
    jax_states_gda = pjitted_identity(jax_states)

  checkpoints.save_checkpoint(
      jax_states_gda, pax_model_path,
      checkpoint_type=checkpoints.CheckpointType.GDA)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base-model-path', type=str, required=True)
  parser.add_argument('--pax-model-path', type=str, required=True)
  args = parser.parse_args()

  print(f'os.list paths = {args.base_model_path}')
  os.listdir(args.base_model_path)

  convert(args.base_model_path, args.pax_model_path)
