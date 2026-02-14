import numpy as np
import bpy
from bmesh.types import BMesh

def active_object_is_mesh(context : bpy.types.Context):
    active_object = context.active_object
    is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())
    return is_mesh

def currently_in_object_mode(context : bpy.types.Context):
    return context.mode == "OBJECT"

def is_manifold_trimesh(bm : BMesh):
    if not np.all(np.array([len(face.verts) for face in bm.faces], dtype=np.int32) == 3):
        return False
    return np.all([edge.is_manifold or edge.is_boundary for edge in bm.edges] + [v.is_manifold for v in bm.verts])
