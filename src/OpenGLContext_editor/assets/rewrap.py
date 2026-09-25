"""The texture a reduced asset needs once the one it inherited stops fitting.

A photogrammetry scan arrives unwrapped by the scanner, into thousands of small
charts -- which is the right answer for the dense mesh, because small charts
distort least, and the wrong one for anything reduced from it. A texture
coordinate means something only inside one chart, so once a reduced triangle
covers more surface than a chart holds, its three corners point at unrelated
places in the image and what it draws is the stripe between them. On a dense
scan that begins at triangle counts a game ships.

No decimator mends that: the fault is in the unwrap, not in the surface, and a
reducer can only carry the coordinates it was given. What mends it is the step
every scan pipeline takes.

    from OpenGLContext_editor.assets.rewrap import unwrap, bake

    laid = unwrap(level_positions, level_indices)
    image = bake(laid, source_positions, source_indices, source_uv, source_image)

:func:`unwrap` gives the reduced mesh charts of its own, made of whole
triangles -- so the failure above cannot happen to it at any triangle count.
:func:`bake` then fills that atlas by asking the original what it looked like at
each texel. What comes back is a level a renderer can draw on its own.

A model carries several maps, and where a texel reads from is the same question
for all of them, so :func:`project` asks it once and :func:`sample` spends the
answer per map. A **tangent-space normal map** goes through :func:`sample_normals`
instead: it holds directions relative to the frame the texture coordinates
define, so a fresh unwrap turns that frame and the bytes have to be turned with
it. Copying them across leaves lighting that leans the wrong way, and the image
itself looks perfectly normal.

**Needs** `xatlas <https://pypi.org/project/xatlas/>`_, which is MIT and ships
wheels for CPython and PyPy alike. It is an extra rather than a dependency: this
is a bake step, and nothing a game runs imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from opengl_decimate import certify, topology

__all__ = [
    'Unwrapped',
    'Projection',
    'Frames',
    'unwrap',
    'project',
    'sample',
    'sample_normals',
    'frames',
    'bake',
    'texel_map',
    'DEFAULT_SIZE',
    'BLEED',
]

#: Pixels on a side of a baked atlas, where a caller names none.
DEFAULT_SIZE = 2048

#: Texels of the gutter that are filled from the chart beside them. A renderer
#: filtering near a chart's edge reaches past it, and a gutter left black draws
#: as a dark seam around every chart; spreading the edge colour outwards gives
#: that filtering something of the right colour to find. Four is the usual
#: allowance -- enough for bilinear filtering and a mip level or two.
BLEED = 4


@dataclass(frozen=True)
class Unwrapped:
    """A mesh with an atlas of its own.

    ``positions`` and ``indices`` are the mesh as it must now be drawn: an
    unwrap splits a vertex wherever a chart boundary runs through it, so this is
    not the mesh that went in. ``source`` says which of the caller's vertices
    each new one came from, which is how any attribute the caller holds --
    normals, colours, weights -- follows the split.
    """

    positions: Any
    indices: Any
    uv: Any
    source: Any
    charts: int


@dataclass(frozen=True)
class Projection:
    """Where each texel of a new atlas reads from on the original surface.

    ``spots`` are ``(column, row)`` pairs into an image ``size`` on a side, and
    ``read`` the texture coordinate in the *original* model's atlas that each
    one takes its colour from. Every map the original carries is sampled with
    the same pair.
    """

    spots: Any
    read: Any
    size: int
    #: Which face of the new mesh each texel belongs to and where in it, and the
    #: same for the triangle of the original it reads from. A map held in the
    #: surface's own frame -- a tangent-space normal map -- is turned from the
    #: one frame into the other with these; see :func:`sample_normals`.
    into: Any = None
    into_weights: Any = None
    onto: Any = None
    onto_weights: Any = None


@dataclass(frozen=True)
class Frames:
    """Per-vertex tangent frames: the directions a normal map is read against.

    ``tangent`` runs the way ``u`` does, ``normal`` is the shading normal, and
    ``handed`` is the sign that reconstructs the bitangent as
    ``cross(normal, tangent) * handed`` -- glTF's own arrangement, so a mirrored
    chart keeps its green channel meaning what it says.
    """

    tangent: Any
    normal: Any
    handed: Any


#: How many candidate texels :func:`_rasterise` tests in one batch of faces.
RASTER_BATCH = 1 << 21


def unwrap(positions: Any, indices: Any, size: int = DEFAULT_SIZE) -> Unwrapped:
    """Lay a mesh out in a fresh atlas of whole triangles.

    The surface is **welded first**. A mesh reduced from a scan carries a vertex
    per chart corner of the atlas it inherited, and an unwrapper handed those
    sees a surface torn into thousands of disconnected pieces and lays out
    thousands of charts. Positions say what is actually joined.

    ``size`` is the atlas the packer aims at; the coordinates come back in
    ``[0, 1]`` either way, so it decides how the charts are packed rather than
    what they are.
    """
    import xatlas

    points, belongs = topology.weld_positions(np.asarray(positions, dtype='d'))
    faces = belongs[np.asarray(indices).reshape(-1, 3)]
    # Welding can leave a triangle with a repeated corner, which covers no area
    # and has no chart to be in.
    usable = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[usable]

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(points, dtype='f4'),
        np.ascontiguousarray(faces, dtype=np.uint32),
    )
    packing = xatlas.PackOptions()
    packing.resolution = int(size)
    atlas.generate(pack_options=packing)
    mapping, laid, uv = atlas[0]

    # `mapping` names a welded point; the caller wants one of their own
    # vertices, so take the first that welded onto it.
    first = np.zeros(len(points), dtype=np.int64)
    first[belongs[::-1]] = np.arange(len(belongs), dtype=np.int64)[::-1]
    # The coordinates come back in [0, 1] already; `resolution` decides how
    # finely the charts are packed into that square, not what units they are in.
    return Unwrapped(
        positions=np.ascontiguousarray(points[mapping], dtype='f4'),
        indices=np.ascontiguousarray(laid.reshape(-1), dtype=np.uint32),
        uv=np.ascontiguousarray(uv, dtype='f4'),
        source=first[mapping],
        charts=int(atlas.chart_count),
    )


def texel_map(laid: Unwrapped, size: int = DEFAULT_SIZE) -> tuple[Any, Any]:
    """Which texels the atlas covers, and where on the model each one sits.

    Returns ``(spots, at)``: integer ``(column, row)`` pairs into an image of
    ``size`` on a side, and the position in the mesh's own space that each
    stands for. Rasterising the mesh in *texture* space is what makes a bake a
    lookup rather than a render -- every texel is asked about directly, so
    nothing depends on what a camera happened to see.

    Rows run the way an image's rows run and the way glTF reads a texture
    coordinate: ``v`` of zero is the top. Flipping that is the mistake that
    fills an atlas with real texture in the wrong places.
    """
    spots, at, _face, _within = _rasterise(laid, size)
    return spots, at


def _rasterise(laid: Unwrapped, size: int) -> tuple[Any, Any, Any, Any]:
    """:func:`texel_map`, and which face each texel came from and where in it.

    A tangent frame belongs to the mesh's vertices, so anything read in the
    surface's own frame -- a normal map -- needs the face and the position
    within it, not only the point in space.
    """
    faces = np.asarray(laid.indices).reshape(-1, 3)
    corner_uv = np.asarray(laid.uv, dtype='d')[faces] * size - 0.5
    corner_at = np.asarray(laid.positions, dtype='d')[faces]

    low = np.maximum(np.floor(corner_uv.min(axis=1)).astype(np.int64), 0)
    high = np.minimum(np.ceil(corner_uv.max(axis=1)).astype(np.int64) + 1, size)
    wide = np.maximum(high[:, 0] - low[:, 0], 0)
    tall = np.maximum(high[:, 1] - low[:, 1], 0)
    first = corner_uv[:, 1] - corner_uv[:, 0]
    second = corner_uv[:, 2] - corner_uv[:, 0]
    area = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    # A texel counts as covered when its centre is inside the triangle. A
    # sliver narrower than a texel would then paint nothing at all, so the test
    # is loosened by half a texel's worth of the triangle's own size.
    slack = 0.5 / np.maximum(np.ptp(corner_uv, axis=1).max(axis=1), 1e-12)
    drawn = np.flatnonzero((wide * tall > 0) & (np.abs(area) >= 1e-12))

    spots: list[Any] = []
    places: list[Any] = []
    whose: list[Any] = []
    within: list[Any] = []
    # Every candidate texel of a batch of faces at once, the batch bounded so
    # the arrays stay a few tens of megabytes whatever the mesh.
    counts = (wide * tall)[drawn]
    ends = np.cumsum(counts)
    begin = 0
    while begin < len(drawn):
        stop = int(np.searchsorted(ends, (ends[begin] - counts[begin])
                                   + RASTER_BATCH, side='right'))
        stop = max(stop, begin + 1)
        batch = drawn[begin:stop]
        each = counts[begin:stop]
        face = np.repeat(batch, each)
        offset = np.arange(int(each.sum())) - np.repeat(np.cumsum(each) - each, each)
        column = low[face, 0] + offset % wide[face]
        row = low[face, 1] + offset // wide[face]
        away = np.stack([column, row], axis=1).astype('d') - corner_uv[face, 0]
        along = (away[:, 0] * second[face, 1] - second[face, 0] * away[:, 1]) / area[face]
        across = (first[face, 0] * away[:, 1] - away[:, 0] * first[face, 1]) / area[face]
        weights = np.stack([1.0 - along - across, along, across], axis=1)
        inside = np.all(weights >= -slack[face][:, None], axis=1)
        spots.append(np.stack([column, row], axis=1)[inside])
        places.append(np.einsum('ij,ijk->ik', weights[inside], corner_at[face[inside]]))
        whose.append(face[inside])
        within.append(weights[inside])
        begin = stop
    if not spots:
        empty = np.zeros((0, 2), dtype=np.int64)
        return (
            empty,
            np.zeros((0, 3), dtype='d'),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 3), dtype='d'),
        )
    return (
        np.concatenate(spots),
        np.concatenate(places),
        np.concatenate(whose),
        np.concatenate(within),
    )


def _barycentric(pixels: Any, corners: Any) -> Any:
    """Where each pixel sits in a triangle, as three weights summing to one."""
    first, second = corners[1] - corners[0], corners[2] - corners[0]
    area = first[0] * second[1] - second[0] * first[1]
    if abs(area) < 1e-12:
        return None
    offset = pixels - corners[0]
    along = (offset[:, 0] * second[1] - second[0] * offset[:, 1]) / area
    across = (first[0] * offset[:, 1] - offset[:, 0] * first[1]) / area
    return np.stack([1.0 - along - across, along, across], axis=1)


def bake(
    laid: Unwrapped,
    positions: Any,
    indices: Any,
    uv: Any,
    image: Any,
    size: int = DEFAULT_SIZE,
) -> Any:
    """Fill a fresh atlas with what the original model shows at each texel.

    Every texel of ``laid``'s atlas stands for a place on the reduced surface.
    That place is projected onto the original -- which triangle of it is nearest,
    and where on that triangle -- and the original's own texture coordinate is
    read there and used to sample ``image``. So the new texture holds what the
    old one held, laid out where the new mesh can find it.

    Comes back as ``(size, size, C)`` bytes for a source of ``C`` channels,
    black wherever no chart covers. A model with several maps to carry over
    wants :func:`project` once and :func:`sample` per map.
    """
    return sample(project(laid, positions, indices, uv, size), image)


def project(
    laid: Unwrapped,
    positions: Any,
    indices: Any,
    uv: Any,
    size: int = DEFAULT_SIZE,
) -> Projection:
    """Work out, once, where on the original each texel of the new atlas reads.

    This is the expensive half of a bake -- every texel is projected onto the
    nearest triangle of a mesh that may hold millions -- and it depends on the
    two surfaces alone. A model carrying base colour, normals, roughness and
    occlusion asks the same question four times over, so the answer is a value
    of its own and :func:`sample` spends it on each map in turn.

    What it costs is set by **how far the two surfaces are apart**, more than by
    either one's size. The search is a widening ring through a grid sized to the
    original's triangles, so a texel of a coarse level, sitting further off the
    original than a fine level's does, is found only after more rings -- and a
    ring's cost grows with its radius. Expect the coarse end of a chain to take
    longer per texel than the fine end, on the same number of texels.
    """
    spots, at, into, within = _rasterise(laid, size)
    if not len(spots):
        return Projection(
            spots=spots,
            read=np.zeros((0, 2), dtype='d'),
            size=size,
            into=into,
            into_weights=within,
            onto=into,
            onto_weights=within,
        )
    faces = np.asarray(indices).reshape(-1, 3)
    corners = np.asarray(positions, dtype='d')[faces]
    triangle, landed = certify.nearest_triangle(at, np.asarray(positions, dtype='d'), indices)
    weights = _weights_on(corners[triangle], landed)
    return Projection(
        spots=spots,
        read=np.einsum('ij,ijk->ik', weights, np.asarray(uv, dtype='d')[faces[triangle]]),
        size=size,
        into=into,
        into_weights=within,
        onto=triangle,
        onto_weights=weights,
    )


def sample(shot: Projection, image: Any) -> Any:
    """One of the original's maps, laid out where the new mesh can find it."""
    image = np.asarray(image)
    channels = image.shape[2] if image.ndim > 2 else 1
    canvas = np.zeros((shot.size, shot.size, channels), dtype=image.dtype)
    if not len(shot.spots):
        return canvas
    read = _filtered(image, shot.read)
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        read = np.clip(np.rint(read), limits.min, limits.max)
    canvas[shot.spots[:, 1], shot.spots[:, 0]] = read.astype(image.dtype).reshape(
        len(shot.spots), -1)
    painted = np.zeros((shot.size, shot.size), dtype=bool)
    painted[shot.spots[:, 1], shot.spots[:, 0]] = True
    return _spread(canvas, painted, BLEED)


def frames(positions: Any, indices: Any, uv: Any, normals: Any) -> Frames:
    """The tangent frames a renderer reconstructs for this mesh, per vertex.

    A normal map is decoded against the frame the *shader* builds, so a bake
    that encodes against any other frame is wrong however self-consistent it is.
    The tangents come from the engine's own ``estimate_tangents`` for that
    reason: one construction, one handedness convention, and a normal map baked
    here means what the renderer reads.
    """
    from OpenGLContext.loaders.gltf.meshes import estimate_tangents

    normals = np.asarray(normals, dtype='f4')
    tangents = estimate_tangents(
        np.asarray(positions, dtype='f4'),
        normals,
        np.asarray(uv, dtype='f4'),
        np.asarray(indices).reshape(-1).astype(np.uint32),
    )
    return Frames(
        tangent=np.asarray(tangents[:, :3], dtype='d'),
        normal=np.asarray(normals, dtype='d'),
        handed=np.asarray(tangents[:, 3], dtype='d'),
    )


def _frame_at(held: Frames, faces: Any, which: Any, weights: Any) -> Any:
    """The frame at a point inside a face, the way a fragment gets one.

    Interpolated across the triangle from its corners and made orthonormal
    again afterwards -- which is what the vertex stage and the fragment stage
    between them do, so a texel and the fragment that will read it agree. A
    per-face frame instead would be constant across each triangle and step at
    every edge, and the mesh would draw its own wireframe in the lighting.

    Rows are tangent, bitangent and normal.
    """
    corners = faces[which]
    normal = np.einsum('ic,ick->ik', weights, held.normal[corners])
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    tangent = np.einsum('ic,ick->ik', weights, held.tangent[corners])
    tangent = tangent - normal * np.einsum('ij,ij->i', tangent, normal)[:, None]
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    # A vertex whose uv could say nothing leaves a zero tangent; any direction
    # across the normal will do there, and the axis it leans on least is the one
    # furthest from it.
    spare = np.zeros_like(normal)
    spare[np.arange(len(normal)), np.argmin(np.abs(normal), axis=1)] = 1.0
    tangent = np.where(length > 1e-9, tangent / np.maximum(length, 1e-12), np.cross(normal, spare))
    handed = np.sign(np.einsum('ic,ic->i', weights, held.handed[corners]))
    handed = np.where(handed == 0.0, 1.0, handed)
    return np.stack([tangent, np.cross(normal, tangent) * handed[:, None], normal], axis=1)


def sample_normals(
    shot: Projection, image: Any, onto: Frames, into: Frames, indices: Any, laid_indices: Any
) -> Any:
    """The original's normal map, re-expressed in the new mesh's own frame.

    A tangent-space normal map is not a picture of anything on its own: each
    texel is a direction *relative to the frame the texture coordinates define*
    at that point on the surface. Give the mesh a different unwrap and the frame
    turns, so copying the bytes across keeps the numbers and changes what they
    mean -- lighting that leans the wrong way, and nothing odd about the image
    itself to say so.

    So each texel goes the whole way round: decoded against ``onto``, the
    original's frame where it reads, which gives a direction in the model's own
    space, and encoded again against ``into``, the new mesh's frame where it
    will be read. Both come from :func:`frames`.
    """
    image = np.asarray(image)
    canvas = np.zeros((shot.size, shot.size, 3), dtype=np.uint8)
    # Straight up in tangent space, which is what a texel of untouched surface
    # holds and the right thing for a gutter to say. Black would be a direction
    # pointing back into the surface, and draws as a ring around every chart.
    canvas[:, :] = (128, 128, 255)
    if not len(shot.spots):
        return canvas

    source = _frame_at(onto, np.asarray(indices).reshape(-1, 3), shot.onto, shot.onto_weights)
    target = _frame_at(into, np.asarray(laid_indices).reshape(-1, 3), shot.into, shot.into_weights)

    leaning = _filtered(image, shot.read)[:, :3] / 127.5 - 1.0

    # Rows of a frame are its tangent, bitangent and normal, so the frame times
    # the direction takes it out into the model's space, and the transpose --
    # the frame being orthonormal -- brings it back into the other one.
    world = np.einsum('irk,ir->ik', source, leaning)
    turned = np.einsum('irk,ik->ir', target, world)
    turned = turned / np.maximum(np.linalg.norm(turned, axis=1, keepdims=True), 1e-12)

    painted = np.zeros((shot.size, shot.size), dtype=bool)
    painted[shot.spots[:, 1], shot.spots[:, 0]] = True
    canvas[shot.spots[:, 1], shot.spots[:, 0]] = np.clip(
        np.rint((turned + 1.0) * 127.5), 0, 255
    ).astype(np.uint8)
    return _spread(canvas, painted, BLEED)


def _filtered(image: Any, read: Any) -> Any:
    """``image`` read at texture coordinates ``read``, bilinearly, as float.

    Texel centres are at half-integers, as a renderer samples them, and the
    edges clamp. ``v`` of zero is the top of the image, which is row zero, as
    glTF has it. Where the new atlas is denser than the source, neighbouring
    texels of it read between the source's texels rather than repeating one.
    """
    source = np.asarray(image, dtype='d')
    if source.ndim == 2:
        source = source[:, :, None]
    height, width = source.shape[:2]
    x = np.asarray(read[:, 0], dtype='d') * width - 0.5
    y = np.asarray(read[:, 1], dtype='d') * height - 0.5
    left = np.floor(x)
    top = np.floor(y)
    across = (x - left)[:, None]
    down = (y - top)[:, None]
    x0 = np.clip(left.astype(np.int64), 0, width - 1)
    x1 = np.clip(left.astype(np.int64) + 1, 0, width - 1)
    y0 = np.clip(top.astype(np.int64), 0, height - 1)
    y1 = np.clip(top.astype(np.int64) + 1, 0, height - 1)
    upper = source[y0, x0] * (1.0 - across) + source[y0, x1] * across
    lower = source[y1, x0] * (1.0 - across) + source[y1, x1] * across
    found: Any = upper * (1.0 - down) + lower * down
    return found


def _spread(canvas: Any, painted: Any, reach: int) -> Any:
    """Grow the painted region outwards, so a chart's edge has a margin.

    One texel at a time, each unpainted texel next to a painted one taking that
    colour. A renderer filtering across a chart's edge lands in the gutter, and
    what it finds there should be the chart rather than whatever the atlas was
    cleared to.
    """
    for _ in range(max(0, reach)):
        if painted.all():
            break
        # One ring per pass: the neighbours are read from the region as it stood
        # when the pass began, so a texel filled by this pass does not go on to
        # fill its own neighbour until the next one.
        was, held = painted.copy(), canvas.copy()
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            source = np.roll(was, shift, axis=axis)
            # A roll wraps, and the far edge is not a neighbour of the near one.
            edge: list[Any] = [slice(None), slice(None)]
            edge[axis] = slice(0, 1) if shift > 0 else slice(-1, None)
            source[tuple(edge)] = False
            fill = source & ~painted
            if not np.any(fill):
                continue
            canvas[fill] = np.roll(held, shift, axis=axis)[fill]
            painted |= fill
    return canvas


def _weights_on(corners: Any, points: Any) -> Any:
    """Barycentric weights of a point already known to lie in its triangle."""
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    offset = points - corners[:, 0]
    first_first = np.einsum('ij,ij->i', first, first)
    first_second = np.einsum('ij,ij->i', first, second)
    second_second = np.einsum('ij,ij->i', second, second)
    offset_first = np.einsum('ij,ij->i', offset, first)
    offset_second = np.einsum('ij,ij->i', offset, second)
    area = np.maximum(first_first * second_second - first_second * first_second, 1e-30)
    along = (second_second * offset_first - first_second * offset_second) / area
    across = (first_first * offset_second - first_second * offset_first) / area
    return np.stack([1.0 - along - across, along, across], axis=1)
