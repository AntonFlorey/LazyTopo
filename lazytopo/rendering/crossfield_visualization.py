import os
from pathlib import Path
import math

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
    active_mesh.calc_loop_triangles()
    vertex_positions = []
    triangle_indices = []
    uvs = []

    v_id = 0
    for face in active_mesh.polygons:
        cross_center = face.center + 0.012 * face.normal
        cross_size = math.sqrt(face.area) / 3.0
        cross_a =  Vector(crossfield[face.index])
        cross_b = face.normal.cross(cross_a).normalized()
        quad_0 = cross_center + cross_size * (cross_a + cross_b)
        quad_1 = cross_center + cross_size * (-cross_a + cross_b)
        quad_2 = cross_center + cross_size * (-cross_a - cross_b)
        quad_3 = cross_center + cross_size * (cross_a - cross_b)
        vertex_positions += [quad_0, quad_1, quad_2, quad_3]
        uvs += [(1,1), (0,1), (0,0), (1,0)]
        triangle_indices += [(v_id, v_id + 1, v_id + 2), (v_id, v_id + 2, v_id + 3)]
        v_id += 4

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
    curvature = attribute_helpers.read_numpy_array_from_vector_attribute(active_mesh.attributes, PRINCIPAL_CURVATURE_ATTR_NAME, len(active_mesh.polygons))
    curvature_unambiguity = attribute_helpers.read_numpy_array_from_float_attribute(active_mesh.attributes, PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME, len(active_mesh.polygons))

    # compute batch info
    active_mesh.calc_loop_triangles()
    vertex_positions = []
    triangle_indices = []
    uvs = []

    v_id = 0
    for face in active_mesh.polygons:
        if curvature_unambiguity[face.index] < rendering_props.curvature_threshold:
            continue
        cross_center = face.center + 0.01 * face.normal
        cross_size = math.sqrt(face.area) / 2.5
        cross_a =  Vector(curvature[face.index])
        cross_b = face.normal.cross(cross_a).normalized()
        quad_0 = cross_center + cross_size * (cross_a + cross_b)
        quad_1 = cross_center + cross_size * (-cross_a + cross_b)
        quad_2 = cross_center + cross_size * (-cross_a - cross_b)
        quad_3 = cross_center + cross_size * (cross_a - cross_b)
        vertex_positions += [quad_0, quad_1, quad_2, quad_3]
        uvs += [(1,1), (0,1), (0,0), (1,0)]
        triangle_indices += [(v_id, v_id + 1, v_id + 2), (v_id, v_id + 2, v_id + 3)]
        v_id += 4

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
