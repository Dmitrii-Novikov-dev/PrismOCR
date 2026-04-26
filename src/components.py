# Connected components detection for segmenting individual glyphs from images

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Component:
    pixels: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int


def connected_components(mask: np.ndarray) -> List[Component]:
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    comps = []

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pix = []

            y0 = y1 = y
            x0 = x1 = x

            while stack:
                cy, cx = stack.pop()
                pix.append((cy, cx))

                y0, y1 = min(y0, cy), max(y1, cy)
                x0, x1 = min(x0, cx), max(x1, cx)

                for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            comps.append(Component(
                pixels=np.array(pix),
                bbox=(y0, x0, y1, x1),
                area=len(pix)
            ))

    return comps