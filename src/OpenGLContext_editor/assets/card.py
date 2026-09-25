"""The billboard a plant becomes once its geometry is not worth drawing.

Past a few tens of metres a clump of real blades costs hundreds of triangles to
occupy a dozen pixels, so the cover chain replaces it with a camera-facing card.
The two have to *match*: the handoff is a dither between them over a distance
window, and a card whose colours or silhouette disagree with the geometry shows
as a shimmer right where the eye is drawn. So the card is not authored -- it is
the geometry itself, rendered front-on into a texture.

    from OpenGLContext_editor.assets.card import bake_card

    width = bake_card('fern_02.glb', 'fern_a', texture, 'fern_a_card.png')

``width`` comes back because a card is as wide as the plant reaches: the
billboard turns to face the camera, so it is drawn one unit tall and ``width``
units across, where ``width`` is twice the plant's horizontal radius over its
height -- as wide as the plant looks from whichever side. The picture on it is
the front view, looking along -z, with the whole depth of the plant inside the
view volume. The texture is cut to the card's shape too, up to :data:`WIDEST`,
so a spreading plant does not spend half its resolution being squeezed into a
square and the other half on empty sky.

Needs a GL context, which it opens and closes itself. This is a bake step; the
baked ``.png`` is what ships.
"""
from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

__all__ = ['bake_card', 'CARD_SIZE', 'WIDEST']

#: How many pixels tall a baked card is. A card is a few dozen pixels on
#: screen at the distance it is drawn, so this is headroom for the mip chain
#: rather than detail anyone resolves.
CARD_SIZE = 512

#: How much wider than tall a card may get before the plant is squeezed into it
#: after all. A spreading plant in a square texture spends its horizontal
#: resolution on the squeeze and its vertical on empty sky, so the texture is
#: cut to the plant's own shape -- but a hedge strip is fifteen times wider than
#: it is tall, and a texture that shape is mostly memory nobody looks at.
WIDEST = 4.0

_VERTEX = """#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
uniform mat4 uOrtho;
out vec2 vTexCoord;
out vec3 vNormal;
void main() {
    vTexCoord = aTexCoord;
    vNormal = aNormal;
    gl_Position = uOrtho * vec4(aPosition, 1.0);
}
"""

# Alpha-cut at the same threshold the clump shader uses, so the silhouette the
# card carries is the silhouette the geometry draws. The flat term stands in for
# the per-fragment sun the geometry gets: a card lit as though every leaf faced
# the light reads as a bright lump on a shaded floor.
_FRAGMENT = """#version 330 core
in vec2 vTexCoord;
in vec3 vNormal;
uniform sampler2D uTexture;
uniform float uCutoff;
out vec4 fColor;
void main() {
    vec4 texel = texture(uTexture, vTexCoord);
    if (texel.a < uCutoff) discard;
    float facing = 0.55 + 0.45 * abs(normalize(vNormal).z);
    fColor = vec4(texel.rgb * facing, 1.0);
}
"""


def bake_card(model: str, mesh: int | str, texture: Any, out: str,
              size: int = CARD_SIZE, cutoff: float = 0.33) -> float:
    """Render ``mesh`` of ``model`` front-on to ``out``; return how wide it is.

    ``texture`` is the plant's cutout texture, as a ``PIL.Image`` or a path.
    The returned width is the plant's width over its height, which is what the
    billboard node wants for its quad.
    """
    from OpenGL.GL import (
        GL_ARRAY_BUFFER,
        GL_CLAMP_TO_EDGE,
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_BUFFER_BIT,
        GL_CULL_FACE,
        GL_DEPTH_ATTACHMENT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_COMPONENT24,
        GL_DEPTH_TEST,
        GL_ELEMENT_ARRAY_BUFFER,
        GL_FALSE,
        GL_FLOAT,
        GL_FRAGMENT_SHADER,
        GL_FRAMEBUFFER,
        GL_FRAMEBUFFER_COMPLETE,
        GL_LINEAR,
        GL_LINEAR_MIPMAP_LINEAR,
        GL_RENDERBUFFER,
        GL_RGBA,
        GL_RGBA8,
        GL_STATIC_DRAW,
        GL_TEXTURE0,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TRIANGLES,
        GL_TRUE,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_INT,
        GL_VERTEX_SHADER,
        glActiveTexture,
        glBindBuffer,
        glBindFramebuffer,
        glBindRenderbuffer,
        glBindTexture,
        glBindVertexArray,
        glBufferData,
        glCheckFramebufferStatus,
        glClear,
        glClearColor,
        glDisable,
        glDrawElements,
        glEnable,
        glEnableVertexAttribArray,
        glFramebufferRenderbuffer,
        glFramebufferTexture2D,
        glGenBuffers,
        glGenerateMipmap,
        glGenFramebuffers,
        glGenRenderbuffers,
        glGenTextures,
        glGenVertexArrays,
        glGetUniformLocation,
        glReadPixels,
        glRenderbufferStorage,
        glTexImage2D,
        glTexParameteri,
        glUniform1f,
        glUniform1i,
        glUniformMatrix4fv,
        glUseProgram,
        glVertexAttribPointer,
        glViewport,
    )
    from OpenGL.GL.shaders import compileProgram, compileShader
    from OpenGLContext.scenegraph.vegetation.clumps import load_clump_glb
    from OpenGLContext.testing.glcontext import hidden_window
    from PIL import Image

    points, normals, uvs, indices, embedded = load_clump_glb(model, mesh=mesh)
    image = (embedded if texture is None
             else texture if not isinstance(texture, str)
             else Image.open(texture))
    image = image.convert('RGBA')
    half_width = float(np.hypot(points[:, 0], points[:, 2]).max()) or 1.0
    height = float(points[:, 1].max()) or 1.0
    # The card is as wide as the plant, up to WIDEST; past that the plant is
    # squeezed, exactly as a square card always squeezed it.
    across = 2.0 * half_width / height
    wide = max(int(round(size * min(across, WIDEST))), 1)

    with hidden_window('bake card', size=(64, 64)):
        program = compileProgram(
            compileShader(_VERTEX, GL_VERTEX_SHADER),
            compileShader(_FRAGMENT, GL_FRAGMENT_SHADER))
        frame = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, frame)
        colour = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, colour)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, wide, size, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, colour, 0)
        depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, wide, size)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, depth)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                "the %dx%d card framebuffer cannot be drawn into (status "
                "0x%04x)" % (wide, size, int(status)))
        glViewport(0, 0, wide, size)
        # Clear to nothing at all: what no triangle covers is what the card's
        # alpha has to leave out, and that IS the silhouette.
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)          # leaves are drawn from either side
        glUseProgram(program)
        # x in [-half, half] and y in [0, height] onto the card, looking -z,
        # and z in [-half, half] into the depth range, which holds the whole
        # plant because half_width is its horizontal radius.
        ortho = np.array([[1.0 / half_width, 0, 0, 0],
                          [0, 2.0 / height, 0, -1.0],
                          [0, 0, -1.0 / half_width, 0],
                          [0, 0, 0, 1.0]], np.float32)
        glUniformMatrix4fv(glGetUniformLocation(program, 'uOrtho'), 1, GL_TRUE,
                           ortho)
        glUniform1i(glGetUniformLocation(program, 'uTexture'), 0)
        glUniform1f(glGetUniformLocation(program, 'uCutoff'), float(cutoff))

        skin = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, skin)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image.width, image.height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, np.asarray(image))
        glGenerateMipmap(GL_TEXTURE_2D)
        for name, value in ((GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR),
                            (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                            (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
            glTexParameteri(GL_TEXTURE_2D, name, value)

        vertices = np.concatenate([points, normals, uvs], 1).astype(np.float32)
        array = glGenVertexArrays(1)
        glBindVertexArray(array)
        buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        for location, width, offset in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
            glVertexAttribPointer(location, width, GL_FLOAT, GL_FALSE, 32,
                                  ctypes.c_void_p(offset))
            glEnableVertexAttribArray(location)
        elements = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elements)
        drawn = np.ascontiguousarray(indices, np.uint32)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, drawn.nbytes, drawn,
                     GL_STATIC_DRAW)
        glDrawElements(GL_TRIANGLES, len(drawn), GL_UNSIGNED_INT, None)

        raw = glReadPixels(0, 0, wide, size, GL_RGBA, GL_UNSIGNED_BYTE)

    picture = Image.frombytes('RGBA', (wide, size), raw).transpose(
        Image.Transpose.FLIP_TOP_BOTTOM)
    picture.save(out)
    return 2.0 * half_width / height


def card_coverage(path: str) -> float:
    """How much of a baked card is not transparent, in [0, 1].

    A card that covers almost none of its texture is a plant rendered too small
    to read, and one that covers almost all of it is a plant that overflowed the
    frame -- either way the bake wants looking at rather than shipping.
    """
    from PIL import Image
    return float((np.asarray(Image.open(path).convert('RGBA'))[..., 3] > 10
                  ).mean())
