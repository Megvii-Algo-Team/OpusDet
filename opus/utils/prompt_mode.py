"""Parse OPUS / TRex2 eval ``mode`` strings
(``text_only``, ``visual.G.16``, ``visual.I.present_only``, …).
"""
import os
import re
from typing import List, Optional, Tuple

PROMPT_MODE = ('text_only', 'text_visual', 'visual')
# Trailing ``mode`` segments after the core spec.
PROMPT_MODE_FLAGS = frozenset({'present_only', 'solo'})


def _no_interactive_chunk_enabled() -> bool:
    """When set, pure ``visual.I`` uses full entity list (no default ``present_only``)."""
    return os.getenv('NO_INTERACTIVE_CHUNK', '0') != '0'


def parse_prompt_mode(mode: str) -> Tuple[str, Tuple[str, ...], Optional[int]]:
    """Split ``mode`` into core spec, trailing flags, and optional chunk size.

    Chunk size suffix (before flags, stripped from core):
      - ``.solo`` flag → chunk size ``1``
      - ``.chunk.N`` → chunk size ``N`` (``N`` positive int)

    Examples:
      ``text_visual.G.16.present_only.solo`` → core, (present_only, solo), 1
      ``text_only.chunk.40`` → text_only, (), 40
      ``text_visual.G.16.chunk.5.present_only`` → text_visual.G.16, (present_only,), 5
    """
    if not mode:
        return '', (), None
    parts = mode.split('.')
    flags: List[str] = []
    chunk_size: Optional[int] = None

    while parts:
        if parts[-1] in PROMPT_MODE_FLAGS:
            flags.insert(0, parts.pop())
            continue
        if len(parts) >= 2 and parts[-2] == 'chunk' and parts[-1].isdigit():
            chunk_size = int(parts.pop())
            parts.pop()  # 'chunk'
            continue
        break

    if 'solo' in flags:
        chunk_size = 1

    return '.'.join(parts), tuple(flags), chunk_size


def prompt_mode_core(mode: str) -> str:
    """Core mode without trailing flags or ``.chunk.N`` (``visual.G.16.present_only.solo`` → ``visual.G.16``)."""
    return parse_prompt_mode(mode)[0]


def prompt_mode_base(mode: str) -> str:
    """Leading segment before the first ``.`` (e.g. ``text_visual`` from ``text_visual.G``)."""
    core = prompt_mode_core(mode)
    if not core:
        return ''
    return core.split('.')[0].strip()


def prompt_mode_is_pure_interactive(core: str) -> bool:
    """Pure ``visual.I`` (not ``text_visual.I``)."""
    return prompt_mode_base(core) == 'visual' and 'I' in core.split('.')


def prompt_mode_present_only(mode: str) -> bool:
    """Restrict prompts to image-present classes (eval: from GT labels).

    Enabled by trailing ``.present_only``, or by default for pure ``visual.I``
    unless ``NO_INTERACTIVE_CHUNK`` is set.
    """
    _, flags, _ = parse_prompt_mode(mode)
    if 'present_only' in flags:
        return True
    if prompt_mode_is_pure_interactive(prompt_mode_core(mode)):
        return not _no_interactive_chunk_enabled()
    return False


def prompt_mode_solo(mode: str) -> bool:
    """One isolated prompt group per forward (``chunk`` size 1)."""
    _, flags, chunk_size = parse_prompt_mode(mode)
    return 'solo' in flags or chunk_size == 1


def prompt_mode_chunk_size(mode: str, default: int = -1) -> int:
    """Effective ``test_cfg.chunked_size`` from mode suffix, else *default*.

    ``.solo`` / ``.chunk.1`` → ``1``; ``.chunk.N`` → ``N``.
    Mode suffix overrides CLI ``--chunked-size`` when set.
    """
    _, flags, chunk_size = parse_prompt_mode(mode)
    if 'solo' in flags:
        return 1
    if chunk_size is not None and chunk_size > 0:
        return chunk_size
    return default


def prompt_mode_visual_num(mode: str) -> Optional[int]:
    """Parse trailing ``.N`` on the core spec (``visual.G.16`` -> ``16``)."""
    core = prompt_mode_core(mode)
    match = re.search(r'\.(\d+)$', core)
    return int(match.group(1)) if match else None


def prompt_mode_includes_text(mode: str) -> bool:
    """Whether to run the language model (``text_only`` / ``text_visual``)."""
    return prompt_mode_base(mode) in ('text_only', 'text_visual')
