from typing import List, TypeVar

T = TypeVar('T')


def chunks(lst: List[T], n: int) -> List[List[T]]:
    """Split *lst* into successive chunks of size *n* (last chunk may be shorter)."""
    if n <= 0:
        raise ValueError(f'chunk size must be positive, got {n}')
    all_ = [lst[i:i + n] for i in range(0, len(lst), n)]
    assert sum(len(c) for c in all_) == len(lst)
    return all_
