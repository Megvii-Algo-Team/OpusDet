# -------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2021 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# -------------------------------------------------------------------------

import re
from typing import Dict, Optional

import numpy as np

adjs = [
    'small', 'large', 'clean', 'dirty', 'hard to see', 
    'cartoon', 'simplified', 'cute', 'digital', 'hand-drawn',
    'toy', 'plushie', 'plastic', 'metal', 'embroidered',
    'origami', 'weird'
]
def needs_an(word: str) -> bool:
    word = word.lower().strip()
    vowels = {'a', 'e', 'i', 'o', 'u'}
    # 特例优先：首字母为 u 但发音是 [ju:] 的情况（如 "unicorn"）
    if word.startswith(('uni', 'use', 'user', 'usual')):
        return False
    # 特例：首字母是 h 但不发音的情况（如 "hour"）
    if word.startswith(('honest', 'hour', 'heir', 'honor')):
        return True
    return word[0] in vowels

def augment_phrase(
        text: str, adj_aug: bool = False, article_aug: bool = False) -> str:
    """Light lexical augmentation for a short noun phrase.

    ``adj_aug``: randomly prepend an adjective (training-style diversity).
    ``article_aug``: strip a leading article then randomly re-attach none / the / a|an.
    If both are False, returns ``text`` unchanged.
    """
    if not adj_aug and not article_aug:
        return text
    article_pattern = r'^(?:\W*)(a|an|the)(?=\s+\w)'
    text = re.sub(article_pattern, '', text, count=1, flags=re.IGNORECASE).lstrip()
    if adj_aug and np.random.random() < 0.5:
        adj = np.random.choice(adjs)
        text = f'{adj} {text}'
    art_prefix = ''
    if article_aug and np.random.random() < 0.5:
        if np.random.random() < 0.3:
            art_prefix = 'the'
        else:
            first_word_match = re.search(r'\b([a-zA-Z][a-zA-Z\-]*)', text)
            if first_word_match:
                first_word = first_word_match.group(1).lower()
                art_prefix = 'an' if needs_an(first_word) else 'a'
    text = f'{art_prefix} {text}'.strip()
    return text

multiple_templates = [
    "There is {} in the scene.",
    "a photo of {} in the scene.",
    "itap of {}.",
    "a picture of {}.",
    "an image of {}.",
    "an image showing {}.",
    "a photo of {}.",
    # detection
    "a detected {}",
    "a bounding box of {}",
    "a region containing {}",
    "a centered photo of {}.",
    "a cropped photo of {}.",
    "a close-up photo of {}.",
    # image
    "a tattoo of {}.",
    "graffiti of {}.",
    "a rendering of {}.",
    "a sculpture of {}.",
    "a good photo of {}.",
    "a bad photo of {}.",
    "a bright photo of {}.",
    "a dark photo of {}.",
    "a low resolution photo of {}.",
    "a jpeg corrupted photo of {}.",
    "a blurry photo of {}.",
    "a pixelated photo of {}.",
    "a black and white photo of {}.",
    "a painting of {}.",
]

simple_template = [
    "a photo of {}.",
]

identity_template = [
    "{}",
]

detect_prompt_templates = [
    "Detect {}",
    "Where is the location of {}"
]
generic_prompt_template = [
     "Detect all bjects in the image"
]


def apply_caption_prompt_to_entity(
        word: str,
        caption_prompt: Optional[Dict] = None) -> str:
    """Expand one entity string using Grounding-DINO-style ``caption_prompt``.

    ``caption_prompt`` maps a cleaned entity name to a dict with optional keys
    ``prefix``, ``name``, ``suffix`` — see ``GroundingDINO.to_enhance_text_prompts``.

    If the word is missing from the mapping, returns ``word`` unchanged.
    """
    if not caption_prompt or not isinstance(caption_prompt, dict):
        return word
    if word not in caption_prompt:
        return word
    spec = caption_prompt[word]
    if not isinstance(spec, dict):
        return word
    out = ''
    if spec.get('prefix'):
        out += spec['prefix']
    if spec.get('name'):
        out += spec['name']
    else:
        out += word
    if spec.get('suffix'):
        out += spec['suffix']
    return out


def caption_prompt_from_sample(data_samples) -> Optional[Dict]:
    """Read ``caption_prompt`` from sample field or ``metainfo`` (GDINO parity)."""
    cp = getattr(data_samples, 'caption_prompt', None)
    if cp is not None:
        return cp if isinstance(cp, dict) else None
    meta = getattr(data_samples, 'metainfo', None) or {}
    cp = meta.get('caption_prompt', None)
    return cp if isinstance(cp, dict) else None

