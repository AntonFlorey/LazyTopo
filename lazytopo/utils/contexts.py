from contextlib import contextmanager

import bpy
import bmesh

weird_mode_table = {
    "PAINT_VERTEX" : "VERTEX_PAINT",
    "EDIT_MESH" : "EDIT",
    "PAINT_WEIGHT" : "WEIGHT_PAINT",
    "PAINT_TEXTURE" : "TEXTURE_PAINT"
}

@contextmanager
def enter_object_mode(context : bpy.types.Context):
    prev_mode = None
    if bpy.context.mode != 'OBJECT':
        prev_mode = context.mode
        bpy.ops.object.mode_set(mode="OBJECT")
    try:
        yield None
    finally:
        if prev_mode is not None:
            if prev_mode in weird_mode_table.keys():
                prev_mode = weird_mode_table[prev_mode]
            bpy.ops.object.mode_set(mode=prev_mode)

@contextmanager
def read_mesh_and_bmesh_in_object_mode(context : bpy.types.Context):
    prev_mode = None
    if bpy.context.mode != 'OBJECT':
        prev_mode = context.mode
        bpy.ops.object.mode_set(mode="OBJECT")
    ao = bpy.context.active_object
    mesh = ao.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    try:
        yield mesh, bm
    finally:
        if prev_mode is not None:
            if prev_mode in weird_mode_table.keys():
                prev_mode = weird_mode_table[prev_mode]
            bpy.ops.object.mode_set(mode=prev_mode)
        bm.free()