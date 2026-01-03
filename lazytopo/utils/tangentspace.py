from typing import NamedTuple
import numpy as np

# blender api
from mathutils import Vector
import bpy
from bmesh.types import BMesh

# addon
from .constants import FACE_SPACE_ORIG_ATTR_NAME, FACE_SPACE_X_ATTR_NAME, FACE_SPACE_Y_ATTR_NAME
from ..io.attribute_helpers import create_new_or_overwrite_attribute, read_numpy_array_from_vector_attribute

def any_tangent_basis(normal : Vector):
    """ Returns an arbitrary orthonormal basis of the tangent plane defined by the given normal """

    u = Vector.orthogonal(normal).normalized()
    v = normal.cross(u).normalized()
    return u, v

class TangentSpace(NamedTuple):
    origin : np.array
    x_axis : np.array
    y_axis : np.array

def get_2D_coords_in_tangent_space(point_in_3d, tangent_space : TangentSpace):
    relative_coord = np.asarray(point_in_3d, dtype=np.float64) - tangent_space.origin
    return np.array([np.dot(tangent_space.x_axis, relative_coord), np.dot(tangent_space.y_axis, relative_coord)], dtype=np.float64)

def get_3D_world_coords_from_tangent_space_coords(point_in_2d, tangent_space : TangentSpace):
    return tangent_space.origin + point_in_2d[0] * tangent_space.x_axis+ point_in_2d[1] * tangent_space.y_axis

def load_or_compute_tangent_space_of_all_faces(mesh : bpy.types.Mesh, always_recompute=False):
    n_faces = len(mesh.polygons)
    if FACE_SPACE_ORIG_ATTR_NAME in mesh.attributes and FACE_SPACE_X_ATTR_NAME in mesh.attributes and FACE_SPACE_Y_ATTR_NAME in mesh.attributes:
        if not always_recompute:
            # load stored attributes
            origins = read_numpy_array_from_vector_attribute(mesh.attributes, FACE_SPACE_ORIG_ATTR_NAME, n_faces)
            x_axes = read_numpy_array_from_vector_attribute(mesh.attributes, FACE_SPACE_X_ATTR_NAME, n_faces)
            y_axes = read_numpy_array_from_vector_attribute(mesh.attributes, FACE_SPACE_Y_ATTR_NAME, n_faces)
            return {face_index : TangentSpace(origin=o, x_axis=x, y_axis=y) for face_index, (o, x, y) in enumerate(zip(origins, x_axes, y_axes))}

    # compute tangent space of all faces
    origins = []
    x_axes = []
    y_axes = []
    for face in mesh.polygons:
        assert len(face.vertices) > 0
        x_ax, y_ax = any_tangent_basis(face.normal)
        orig = mesh.vertices[face.vertices[0]].co
        origins += [val for val in orig]
        x_axes += [val for val in x_ax]
        y_axes += [val for val in y_ax]
    
    origins_attribute = create_new_or_overwrite_attribute(mesh.attributes, FACE_SPACE_ORIG_ATTR_NAME, "FLOAT_VECTOR", "FACE")
    origins_attribute.data.foreach_set("vector", origins)
    x_ax_attribute = create_new_or_overwrite_attribute(mesh.attributes, FACE_SPACE_X_ATTR_NAME, "FLOAT_VECTOR", "FACE")
    x_ax_attribute.data.foreach_set("vector", x_axes)
    y_ax_attribute = create_new_or_overwrite_attribute(mesh.attributes, FACE_SPACE_Y_ATTR_NAME, "FLOAT_VECTOR", "FACE")  
    y_ax_attribute.data.foreach_set("vector", y_axes)
    
    return {face_index : TangentSpace(origin=o, x_axis=x, y_axis=y) for face_index, (o, x, y) in enumerate(zip(np.asarray(origins, dtype=np.float32).reshape((n_faces ,3)), 
                                                                                                             np.asarray(x_axes, dtype=np.float32).reshape((n_faces, 3)), 
                                                                                                             np.asarray(y_axes, dtype=np.float32).reshape((n_faces, 3))))}
