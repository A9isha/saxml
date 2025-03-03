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
"""Tests for lm_tokenizer."""

import os

from absl import flags
from praxis import base_layer
from praxis import pax_fiddle
from saxml.server.pax.lm import lm_tokenizer
import tensorflow as tf

instantiate = base_layer.instantiate
FLAGS = flags.FLAGS


def _CreateParams():
  p = pax_fiddle.Config(
      lm_tokenizer.LMTokenizer,
      # From https://github.com/google/sentencepiece/tree/master/python:
      #   <unk>:  0
      #   <s>:    1
      #   </s>:   2
      #   _He:    151
      #   ll:     88
      #   o:      21
      #   _world: 887
      spm_model=os.path.join(
          FLAGS.test_srcdir,
          '__main__/saxml/server/pax/lm/test_data',
          'test_model.model',
      ),
      target_sos_id=0,
      target_eos_id=1,
  )
  return p


def _CreateTokenizedParams():
  p = pax_fiddle.Config(
      lm_tokenizer.LMTokenizer,
      # From https://github.com/google/sentencepiece/tree/master/python:
      #   <unk>:  0
      #   <s>:    1
      #   </s>:   2
      #   _He:    151
      #   ll:     88
      #   o:      21
      #   _world: 887
      spm_model=os.path.join(
          FLAGS.test_srcdir,
          '__main__/saxml/server/pax/lm/test_data',
          'test_model.model',
      ),
      target_sos_id=0,
      target_eos_id=50256,
      tokenized_input=True,
      tokenized_output=True,
      eos_padding_and_no_sos=True,
  )
  return p


class LMTokenizerTest(tf.test.TestCase):

  def testEmptyStringsToIds(self):
    p = _CreateParams()
    tokenizer = p.Instantiate()
    max_length = 5
    strs = ['', '']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], ids)
    self.assertAllEqual([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], labels)
    self.assertAllEqual(
        [[0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0]], paddings
    )

  def testStringsToIds(self):
    p = _CreateParams()
    tokenizer = p.Instantiate()
    max_length = 5
    strs = ['Hello', 'world']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 151, 88, 21, 0], [0, 887, 0, 0, 0]], ids)
    self.assertAllEqual([[151, 88, 21, 1, 0], [887, 1, 0, 0, 0]], labels)
    self.assertAllEqual(
        [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0]], paddings
    )

  def testStringsToIdsSliceLeft(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    max_length = 3
    strs = ['Hello', 'world']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 151, 88], [0, 887, 0]], ids)
    self.assertAllEqual([[151, 88, 1], [887, 1, 0]], labels)
    self.assertAllEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], paddings)

  def testStringsToIdsSliceRight(self):
    p = _CreateParams()
    p.slice_left = False
    tokenizer = instantiate(p)
    max_length = 3
    strs = ['Hello', 'world']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 88, 21], [0, 887, 0]], ids)
    self.assertAllEqual([[88, 21, 1], [887, 1, 0]], labels)
    self.assertAllEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], paddings)

  def testIdsToStrings(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    ids = [[151, 88, 21, 0, 0], [887, 0, 0, 0, 0]]
    strs = tokenizer.IdsToStrings(ids)
    # This SPM doesn't understand padding so manually truncate.
    strs = [tf.strings.substr(s, 0, 5) for s in strs]
    self.assertEqual([b'Hello', b'world'], strs)

  def testEmptyTokenizedStringsToIds(self):
    p = _CreateTokenizedParams()
    tokenizer = p.Instantiate()
    max_length = 5
    strs = ['', '']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual(
        [
            [50256, 50256, 50256, 50256, 50256],
            [50256, 50256, 50256, 50256, 50256],
        ],
        ids,
    )
    self.assertAllEqual(
        [
            [50256, 50256, 50256, 50256, 50256],
            [50256, 50256, 50256, 50256, 50256],
        ],
        labels,
    )
    self.assertAllEqual(
        [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]], paddings
    )

  def testTokenizedStringsToIds(self):
    p = _CreateTokenizedParams()
    tokenizer = p.Instantiate()
    max_length = 5
    strs = ['151,88,21', '887']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual(
        [[151, 88, 21, 50256, 50256], [887, 50256, 50256, 50256, 50256]], ids
    )
    self.assertAllEqual(
        [[151, 88, 21, 50256, 50256], [887, 50256, 50256, 50256, 50256]], labels
    )
    self.assertAllEqual(
        [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0]], paddings
    )

  def testTokenizedStringsToIdsSliceLeft(self):
    p = _CreateTokenizedParams()
    tokenizer = instantiate(p)
    max_length = 2
    strs = tf.ragged.constant(['151,88,21', '887'])
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[151, 88], [887, 50256]], ids)
    self.assertAllEqual([[151, 88], [887, 50256]], labels)
    self.assertAllEqual([[0.0, 0.0], [0.0, 1.0]], paddings)

  def testTokenizedStringsToIdsSliceRight(self):
    p = _CreateTokenizedParams()
    p.slice_left = False
    tokenizer = instantiate(p)
    max_length = 2
    strs = ['151,88,21', '887']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[88, 21], [887, 50256]], ids)
    self.assertAllEqual([[88, 21], [887, 50256]], labels)
    self.assertAllEqual([[0.0, 0.0], [0.0, 1.0]], paddings)

  def testIdsToTokenizedStrings(self):
    p = _CreateTokenizedParams()
    tokenizer = instantiate(p)
    ids = [[151, 88, 21, 0, 0], [887, 0, 0, 0, 0]]
    strs = tokenizer.IdsToStrings(ids)
    self.assertAllEqual([b'151,88,21,0,0', b'887,0,0,0,0'], strs.numpy())


if __name__ == '__main__':
  tf.test.main()
