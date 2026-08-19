"""Raising and lowering ground by hand.

A designer who needs a hill where the generator put a plain does not want a
different landscape; they want *that* landscape with a hill on it. A
:class:`SculptStroke` is one gesture of a brush recorded as data -- where, how
wide, how much, and how rough -- on the edit stack of a
:class:`~OpenGLContext_editor.world.height.HeightSource`.

Every stroke is bounded and vectorised, so a landscape carrying a hundred of
them costs a bake a hundred rectangle comparisons and arithmetic over the few
samples inside each.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from OpenGLContext.loaders.tiles3d.procedural import fbm

from OpenGLContext_editor.world.height import HeightEdit, register_edit

__all__ = ['SculptStroke']

#: How far apart the fractal detail's features are, as a fraction of the
#: brush's radius. A quarter: coarse enough to read as shape rather than as
#: noise, fine enough that a raised hill is not a smooth dome.
DETAIL_SCALE = 0.25


@register_edit
@dataclass(frozen=True)
class SculptStroke(HeightEdit):
    """One gesture of the brush: ground raised or lowered about a point.

    ``amount`` is metres at the centre, positive up. ``falloff`` is how sharply
    the lift dies away towards ``radius``: 1 is a broad shoulder, higher is a
    tighter dome, and it reaches zero *at* the radius either way, so a stroke
    never steps.

    ``detail`` is how much of the lift is fractal variation rather than a
    smooth dome, from 0 to about 1. It is a share of the lift, not a free
    amount, so a gentle stroke gets gentle detail; a raised hill then sits in
    the same visual family as the land around it instead of reading as a bubble
    on the map.
    """

    KIND: ClassVar[str] = 'sculpt'
    centre: tuple[float, float] = (0.0, 0.0)
    radius: float = 100.0
    amount: float = 20.0
    falloff: float = 2.0
    detail: float = 0.35
    #: Which variation this stroke gets. Recorded, so the same stroke makes the
    #: same ground when the project is opened again.
    seed: int = 0

    def bounds(self) -> tuple[float, float, float, float]:
        x, z = self.centre
        reach = abs(float(self.radius))
        return (x - reach, z - reach, x + reach, z + reach)

    def profile(self, distance: Any) -> np.ndarray:
        """How much of the stroke reaches each distance from its centre.

        One at the centre, zero at the radius and outside it, and smooth at
        both ends: a brush whose edge is a step leaves a cliff a road cannot
        cross, and one whose middle is a point leaves a spike.
        """
        reach = max(abs(float(self.radius)), 1e-9)
        near = np.clip(1.0 - np.asarray(distance, dtype='d') / reach, 0.0, 1.0)
        return np.asarray(near ** max(float(self.falloff), 1e-6), dtype='d')

    def delta(self, x: Any, z: Any, height: Any) -> np.ndarray:
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        reach = self.profile(np.hypot(x - self.centre[0], z - self.centre[1]))
        lift = float(self.amount) * reach
        if not self.detail:
            return lift
        # Modulating the lift rather than adding to it is what keeps the detail
        # inside the brush: at the rim there is no lift, so there is nothing to
        # vary, and the stroke still meets the land it was drawn on.
        grain = max(abs(float(self.radius)) * DETAIL_SCALE, 1e-6)
        noise = np.asarray(fbm(x / grain, z / grain,
                               seed=101 + int(self.seed) * 7, octaves=4),
                           dtype='d') - 0.5
        varied: np.ndarray = lift * (1.0 + 2.0 * float(self.detail) * noise)
        return varied

    def to_json(self) -> dict[str, Any]:
        return {'kind': self.KIND,
                'centre': [float(self.centre[0]), float(self.centre[1])],
                'radius': float(self.radius), 'amount': float(self.amount),
                'falloff': float(self.falloff), 'detail': float(self.detail),
                'seed': int(self.seed)}

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> SculptStroke:
        centre = document.get('centre', (0.0, 0.0))
        return cls(centre=(float(centre[0]), float(centre[1])),
                   radius=float(document.get('radius', 100.0)),
                   amount=float(document.get('amount', 20.0)),
                   falloff=float(document.get('falloff', 2.0)),
                   detail=float(document.get('detail', 0.35)),
                   seed=int(document.get('seed', 0)))
