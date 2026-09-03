from collections import OrderedDict
from typing import Collection, List, Optional, Sequence

import numpy as np


class LRUCache:
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def keys(self):
        return self.cache.keys()

    def has(self, key) -> bool:
        return key in self.cache

    def ordered_keys(self, exclude: Optional[Collection] = None) -> List:
        """LRU order: oldest (cold) first, newest (hot / MRU) last."""
        if exclude is None:
            return list(self.cache.keys())
        exclude_set = set(exclude)
        return [k for k in self.cache.keys() if k not in exclude_set]

    def lru_groups(
            self,
            exclude: Optional[Collection] = None,
            n_groups: int = 3) -> List[List]:
        """Split cached keys into ``n_groups`` by LRU rank (cold → hot)."""
        keys = self.ordered_keys(exclude)
        n = len(keys)
        if n == 0 or n_groups <= 0:
            return [[] for _ in range(max(n_groups, 0))]
        n_groups = min(n_groups, n)
        cuts = [int(n * i / n_groups) for i in range(n_groups + 1)]
        return [keys[cuts[i]:cuts[i + 1]] for i in range(n_groups)]

    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key, value) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def pop(self, key, value):
        self.cache.pop(key, None)


_NEG_LRU_GROUPS = 3
_NEG_GROUP_QUOTAS = (0.2, 0.3, 0.5)
_NEG_MIN_POOL_FOR_GROUPS = 30


def _sample_lru_grouped_negatives(
        groups: Sequence[Sequence],
        k: int,
        quotas: Sequence[float],
) -> List:
    """Sample ``k`` negatives: more from hot (last) LRU groups by ``quotas``.

    ``groups[0]`` = cold (LRU head), ``groups[-1]`` = hot (MRU tail).
    Within each group, sampling is uniform.
    """
    if k <= 0:
        return []
    groups = [list(g) for g in groups]
    n_g = len(groups)
    if n_g == 0:
        return []
    if len(quotas) != n_g:
        raise ValueError(
            f'quotas length {len(quotas)} != n_groups {n_g}')

    pool = [x for g in groups for x in g]
    if not pool:
        return []
    k = min(k, len(pool))

    q = list(quotas)
    q_sum = sum(q)
    if q_sum <= 0:
        q = [1.0 / n_g] * n_g
    else:
        q = [x / q_sum for x in q]

    counts = [0] * n_g
    for i in range(n_g):
        if groups[i]:
            counts[i] = int(k * q[i])
    rem = k - sum(counts)
    # remainder → hotter groups first
    for i in range(n_g - 1, -1, -1):
        if rem <= 0:
            break
        if groups[i]:
            counts[i] += 1
            rem -= 1

    picked: List = []
    used = set()
    for i, g in enumerate(groups):
        if counts[i] <= 0 or not g:
            continue
        avail = [x for x in g if x not in used]
        ni = min(counts[i], len(avail))
        if ni <= 0:
            continue
        sel = np.random.choice(
            avail, size=ni, replace=False).tolist()
        picked.extend(sel)
        used.update(sel)

    if len(picked) < k:
        rest = [x for x in pool if x not in used]
        need = min(k - len(picked), len(rest))
        if need > 0:
            extra = np.random.choice(
                rest, size=need, replace=False).tolist()
            picked.extend(extra)
    return picked[:k]


def cache_and_sample_negative_labels(
        cache: Optional[LRUCache],
        labels,
        num_negatives: int = 10,
        sample: str = 'random',
) -> List:
    """Update LRU with prompt entities, return negative label samples.

    Args:
        cache: Label memory bank; ``None`` skips update and returns ``[]``.
        labels: Labels to insert into the bank and exclude when sampling
            negatives (typically all entities already in the current prompt).
        num_negatives: Target count. ``<= 0`` only updates the bank.
        sample: ``'random'`` = uniform over the bank; ``'hot'`` = LRU-grouped
            sampling biased toward recently seen labels (when pool is large
            enough).
    """
    if cache is None:
        return []

    pos_set = set(labels)
    for label in labels:
        cache.put(label, label)

    if num_negatives <= 0:
        return []

    neg_pool = cache.ordered_keys(exclude=pos_set)
    if not neg_pool:
        return []

    k = min(max(num_negatives, 1), len(neg_pool))

    if (sample == 'hot'
            and len(neg_pool) >= _NEG_MIN_POOL_FOR_GROUPS
            and _NEG_LRU_GROUPS > 1):
        groups = cache.lru_groups(exclude=pos_set, n_groups=_NEG_LRU_GROUPS)
        return _sample_lru_grouped_negatives(
            groups, k, quotas=_NEG_GROUP_QUOTAS)

    return np.random.choice(neg_pool, size=k, replace=False).tolist()
