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

def compute_cross_batch_data(mesh : bpy.types.Mesh, crosses, index_filter = None):
    print("called new function!")
    n_crosses = len(mesh.polygons) if index_filter is None else len(index_filter) 

    print("getting mesh data...")
    start_time = time.time()

    face_normals = np.asarray([face.normal for face in mesh.polygons])    
    cross_centers = np.asarray([face.center for face in mesh.polygons]) + (0.01 * face_normals)
    cross_sizes = np.sqrt(np.asarray([face.area for face in mesh.polygons])) / 3.0
    if index_filter is not None:
        face_normals = face_normals[index_filter, :]
        cross_centers = cross_centers[index_filter, :]
        cross_sizes = cross_sizes[index_filter]
        crosses = crosses[index_filter, :]
    print("done after", time.time() - start_time, "seconds")

    print("rotating crosses by 90 degree...")
    start_time = time.time()
    rotated_crosses_90deg = np.cross(face_normals, crosses, axis=1)
    print("done after", time.time() - start_time, "seconds")

    print("computing einsums...")
    start_time = time.time()
    upper_right = np.einsum("i,ij->ij", cross_sizes, crosses + rotated_crosses_90deg)
    upper_left = np.einsum("i,ij->ij", cross_sizes, rotated_crosses_90deg - crosses)
    print("done after", time.time() - start_time, "seconds")

    print("computing quads...")
    start_time = time.time()
    quads_0 = cross_centers + upper_right
    quads_1 = cross_centers + upper_left
    quads_2 = cross_centers - upper_right
    quads_3 = cross_centers - upper_left
    print("done after", time.time() - start_time, "seconds")

    print("creating the final data...")
    start_time = time.time()
    vertex_positions = np.concatenate([quads_0, quads_1, quads_2, quads_3], dtype=np.float32)
    print("positions concatenated after", time.time() - start_time, "seconds")
    uvs = np.repeat([(1, 1), (0, 1), (0, 0), (1, 0)], n_crosses, axis=0).astype(np.float32)
    print("uvs done after", time.time() - start_time, "seconds")
    f_idx = np.arange(n_crosses, dtype=np.int32)
    offsets_a = np.array([0, n_crosses, 2 * n_crosses], dtype=np.int32)
    offsets_b = np.array([0, 2 * n_crosses, 3 * n_crosses], dtype=np.int32)
    triangle_indices_a = f_idx[:, None] + offsets_a
    triangle_indices_b = f_idx[:, None] + offsets_b
    triangle_indices = np.concatenate([triangle_indices_a, triangle_indices_b], dtype=np.int32)
    print("triangle indices done after", time.time() - start_time, "seconds")

    return vertex_positions, uvs, triangle_indices

def update_crossfield_visualization(self, context : bpy.types.Context):
    global _crossfield_drawing_handle
    global _crossfield_batch
    global _crossfield_texture
    rendering_props = context.scene.lazytopo_settings
    ao : bpy.types.Object = context.active_object
    # remove old crossfield drawings
    hide_crossfield()

    if not rendering_props.show_crossfield or ao.type != "MESH":
        return redraw_view_3d(context)
    
    active_mesh : bpy.types.Mesh = ao.data
    if not CROSSFIELD_ATTR_NAME in active_mesh.attributes:
        return redraw_view_3d(context)
    
    # load the crossfield
    crossfield = attribute_helpers.read_numpy_array_from_vector_attribute(active_mesh.attributes, CROSSFIELD_ATTR_NAME, len(active_mesh.polygons))

    # compute batch info
    vertex_positions, uvs, triangle_indices = compute_cross_batch_data(active_mesh, crossfield)
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

def update_curvature_visualization(self, context : bpy.types.Context):
    global _curvature_drawing_handle
    global _curvature_batch
    global _curvature_texture
    rendering_props = context.scene.lazytopo_settings
    ao : bpy.types.Object = context.active_object
    # remove old crossfield drawings
    hide_curvature()

    if ao.type != "MESH":
        return redraw_view_3d(context)
    
    active_mesh : bpy.types.Mesh = ao.data
    if not (PRINCIPAL_CURVATURE_ATTR_NAME in active_mesh.attributes and PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME in active_mesh.attributes):
        return redraw_view_3d(context)
    
    # load the directions
    
    curvature_unambiguity = attribute_helpers.read_numpy_array_from_float_attribute(active_mesh.attributes, PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME, len(active_mesh.polygons))
    crosses_to_display = [f_idx for f_idx in range(len(active_mesh.polygons)) if curvature_unambiguity[f_idx] >= rendering_props.curvature_threshold]
    curvature = attribute_helpers.read_numpy_array_from_vector_attribute(active_mesh.attributes, PRINCIPAL_CURVATURE_ATTR_NAME, len(active_mesh.polygons))
    n_crosses = len(crosses_to_display)
    if n_crosses == 0:
        return

    # compute batch info
    vertex_positions, uvs, triangle_indices = compute_cross_batch_data(active_mesh, curvature, crosses_to_display)

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
    update_curvature_visualization(self, context)
    update_crossfield_visualization(self, context)
