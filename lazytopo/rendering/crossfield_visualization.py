import os
from pathlib import Path
import math
import numpy as np
import time

from mathutils import Vector
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .rendering_helpers import deactivate_draw_callback, redraw_view_3d
from ..io import attribute_helpers
from ..utils.constants import CROSSFIELD_ATTR_NAME, PRINCIPAL_CURVATURE_ATTR_NAME, PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME
from ..utils.contexts import enter_object_mode

_crossfield_drawing_handle = None
_crossfield_batch = None
_crossfield_texture = None

_curvature_drawing_handle = None
_curvature_batch = None
_curvature_texture = None

def hide_curvature():
    global _curvature_drawing_handle
    deactivate_draw_callback(_curvature_drawing_handle)
    _curvature_drawing_handle = None

def hide_crossfield():
    global _crossfield_drawing_handle
    deactivate_draw_callback(_crossfield_drawing_handle)
    _crossfield_drawing_handle = None

def crosses_drawing_callback(batch : gpu.types.GPUBatch, texture : gpu.types.GPUTexture):
    shader = gpu.shader.from_builtin('IMAGE')   
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.face_culling_set('BACK')
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)

def read_face_data_for_crossfield_visualization(mesh : bpy.types.Mesh):
    n_faces = len(mesh.polygons)

    area_buffer = np.zeros(len(mesh.polygons), np.float32)
    mesh.polygons.foreach_get("area", area_buffer)

    normals_buffer = np.zeros(n_faces * 3, dtype=np.float32)
    mesh.polygons.foreach_get("normal", normals_buffer)

    centers_buffer = np.zeros(n_faces * 3, dtype=np.float32)
    mesh.polygons.foreach_get("center", centers_buffer)

    return area_buffer, normals_buffer.reshape((n_faces, 3)), centers_buffer.reshape((n_faces, 3))

def compute_cross_batch_data(mesh : bpy.types.Mesh, crossfield, face_areas, face_normals, face_centers, cross_size_quotient = 3.0):
    n_crosses = crossfield.shape[0]

    cross_sizes = np.sqrt(face_areas) / cross_size_quotient
    cross_centers = face_centers + (0.01 * face_normals)
    rotated_crosses_90deg = np.cross(face_normals, crossfield, axis=1)
    upper_right = np.einsum("i,ij->ij", cross_sizes, crossfield + rotated_crosses_90deg)
    upper_left = np.einsum("i,ij->ij", cross_sizes, rotated_crosses_90deg - crossfield)
    quads_0 = cross_centers + upper_right
    quads_1 = cross_centers + upper_left
    quads_2 = cross_centers - upper_right
    quads_3 = cross_centers - upper_left

    vertex_positions = np.concatenate([quads_0, quads_1, quads_2, quads_3], dtype=np.float32)
    uvs = np.repeat([(1, 1), (0, 1), (0, 0), (1, 0)], n_crosses, axis=0).astype(np.float32)
    f_idx = np.arange(n_crosses, dtype=np.int32)
    triangle_indices_a = f_idx[:, None] + np.array([0, n_crosses, 2 * n_crosses], dtype=np.int32)
    triangle_indices_b = f_idx[:, None] + np.array([0, 2 * n_crosses, 3 * n_crosses], dtype=np.int32)
    triangle_indices = np.concatenate([triangle_indices_a, triangle_indices_b], dtype=np.int32)

    return vertex_positions, uvs, triangle_indices

def update_crossfield_visualization(self, context : bpy.types.Context, precomputed_face_data = None):
    global _crossfield_drawing_handle
    global _crossfield_batch
    global _crossfield_texture
    crossfield_settings = context.scene.lazytopo_crossfield_settings
    ao : bpy.types.Object = context.active_object
    # remove old crossfield drawings
    hide_crossfield()

    if not crossfield_settings.show_crossfield or ao.type != "MESH":
        return redraw_view_3d(context)
    
    with enter_object_mode(context):
        active_mesh : bpy.types.Mesh = ao.data
        if not CROSSFIELD_ATTR_NAME in active_mesh.attributes:
            return redraw_view_3d(context)
        
        # get face data if missing
        if precomputed_face_data is None:
            precomputed_face_data = read_face_data_for_crossfield_visualization(active_mesh)

        # load the crossfield
        crossfield = attribute_helpers.read_numpy_array_from_vector_attribute(active_mesh.attributes, CROSSFIELD_ATTR_NAME, len(active_mesh.polygons))

        # compute batch info
        vertex_positions, uvs, triangle_indices = compute_cross_batch_data(active_mesh, crossfield, *precomputed_face_data)
        n_faces = len(active_mesh.polygons)

    assert triangle_indices.shape == (2*n_faces, 3), f"Wrong triangle ids shape: {triangle_indices.shape}. Expected: {(2*n_faces, 3)}"
    assert uvs.shape == (4*n_faces, 2), f"Wrong uvs shape: {uvs.shape}. Expected: {(4*n_faces, 2)}"
    assert vertex_positions.shape == (4*n_faces, 3), f"Wrong v-pos shape: {vertex_positions.shape}. Expected: {(4*n_faces, 3)}"

    # make batch
    shader = gpu.shader.from_builtin('IMAGE')
    _crossfield_batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions, "texCoord": uvs}, indices=triangle_indices)

    # load image
    _crossfield_texture = gpu.texture.from_image(bpy.data.images.load(filepath=str(Path(os.path.realpath(__file__)).parent.parent / "textures/BlackCross.png"), check_existing=True))
    
    # set the draw handler
    _crossfield_drawing_handle = bpy.types.SpaceView3D.draw_handler_add(crosses_drawing_callback, (_crossfield_batch, _crossfield_texture), "WINDOW", "POST_VIEW")
    redraw_view_3d(context)

def update_curvature_visualization(self, context : bpy.types.Context, precomputed_face_data):
    global _curvature_drawing_handle
    global _curvature_batch
    global _curvature_texture
    crossfield_settings = context.scene.lazytopo_crossfield_settings
    ao : bpy.types.Object = context.active_object
    # remove old crossfield drawings
    hide_curvature()

    if not crossfield_settings.show_minmax_curvature or ao.type != "MESH":
        return redraw_view_3d(context)
    
    with enter_object_mode(context):
        active_mesh : bpy.types.Mesh = ao.data
        if not (PRINCIPAL_CURVATURE_ATTR_NAME in active_mesh.attributes and PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME in active_mesh.attributes):
            return redraw_view_3d(context)
        
        # get face data if missing
        if precomputed_face_data is None:
            precomputed_face_data = read_face_data_for_crossfield_visualization(active_mesh)

        # load the directions
        curvature_unambiguity = attribute_helpers.read_numpy_array_from_float_attribute(active_mesh.attributes, PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME, len(active_mesh.polygons))
        id_mask = [f_idx for f_idx in range(len(active_mesh.polygons)) if curvature_unambiguity[f_idx] >= crossfield_settings.curvature_threshold]
        curvature = attribute_helpers.read_numpy_array_from_vector_attribute(active_mesh.attributes, PRINCIPAL_CURVATURE_ATTR_NAME, len(active_mesh.polygons))
        n_crosses = len(id_mask)
        if n_crosses == 0:
            return

        # compute batch info
        vertex_positions, uvs, triangle_indices = compute_cross_batch_data(active_mesh, 
                                                                        curvature[id_mask,:], 
                                                                        precomputed_face_data[0][id_mask],
                                                                        precomputed_face_data[1][id_mask,:],
                                                                        precomputed_face_data[2][id_mask,:],
                                                                        2.5)

    assert triangle_indices.shape == (2*n_crosses, 3), f"Wrong triangle ids shape: {triangle_indices.shape}. Expected: {(2*n_crosses, 3)}"
    assert uvs.shape == (4*n_crosses, 2), f"Wrong uvs shape: {uvs.shape}. Expected: {(4*n_crosses, 2)}"
    assert vertex_positions.shape == (4*n_crosses, 3), f"Wrong v-pos shape: {vertex_positions.shape}. Expected: {(4*n_crosses, 3)}"

    # make batch
    shader = gpu.shader.from_builtin('IMAGE')
    _curvature_batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions, "texCoord": uvs}, indices=triangle_indices)

    # load image
    _curvature_texture = gpu.texture.from_image(bpy.data.images.load(filepath=str(Path(os.path.realpath(__file__)).parent.parent / "textures/BlueCross.png"), check_existing=True))
    
    # set the draw handler
    _curvature_drawing_handle = bpy.types.SpaceView3D.draw_handler_add(crosses_drawing_callback, (_curvature_batch, _curvature_texture), "WINDOW", "POST_VIEW")
    redraw_view_3d(context)

def update_all_crosses(self, context : bpy.types.Context):
    precomputed_face_data = read_face_data_for_crossfield_visualization(context.active_object.data)
    update_curvature_visualization(self, context, precomputed_face_data)
    update_crossfield_visualization(self, context, precomputed_face_data)
