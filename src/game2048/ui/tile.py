from __future__ import annotations

from html import escape


def tile_label(exponent: int) -> str:
    return "" if exponent == 0 else f"{2**exponent:,}"


def render_tile(exponent: int, index: int) -> str:
    label = tile_label(exponent)
    aria_label = "Empty tile" if exponent == 0 else f"Tile {label}"
    return (
        f'<div class="tile" data-exp="{exponent}" data-tile-index="{index}" '
        f'aria-label="{escape(aria_label)}">{escape(label)}</div>'
    )
