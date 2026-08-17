"""Baking an authored world into a streamable 3D Tiles octree.

The pipeline is four steps, and each is usable on its own:

``bounds``    the axis-aligned box the baker measures in, and the Z-up volume a
              tileset states
``octree``    the spatial partition: a world's content divided into nodes with a
              geometric error per level
``layers``    what a node's content is made of -- a heightfield sampled at the
              node's resolution, instances placed within its bounds
``driver``    the bake itself: walk the tree, write a ``.glb`` per node through
              the engine's glTF writer, and emit ``tileset.json``

The output is what ``OpenGLContext.loaders.tiles3d`` streams, so a baked world
is loaded by the same runtime as any other 3D Tiles dataset.
"""
