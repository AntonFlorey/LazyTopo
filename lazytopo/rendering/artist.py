import numpy.typing
import typing
import numpy as np
from itertools import chain

import bpy
import gpu
import gpu_extras.batch

from .rendering_helpers import deactivate_draw_callback, redraw_view_3d
from . import shaders

class TriangleRenderData:
    def __init__(self, dim : str):
        self.vertex_coords : numpy.typing.NDArray = np.empty(shape=((0,3) if dim == "3D" else (0,2)), dtype=np.float32)
        self.triangle_indices : numpy.typing.NDArray = np.empty(shape=(0,3), dtype=np.int32)
        self.vertex_colors : numpy.typing.NDArray = np.empty(shape=(0,4), dtype=np.float32)
        self.draw_batch : gpu.types.GPUBatch = None
        self.dirty : bool = False
        self.shader : gpu.types.GPUShader = gpu.shader.from_builtin("SMOOTH_COLOR")

    def create_fresh_batch(self):
        self.draw_batch = gpu_extras.batch.batch_for_shader(self.shader, 'TRIS', {"pos" : self.vertex_coords, "color" : self.vertex_colors}, indices=self.triangle_indices)
        self.dirty = False

    def draw_callback(self):
        if self.draw_batch == None:
            return
        self.shader.bind()
        self.draw_batch.draw(self.shader)   

class LineRenderData:
    def __init__(self, dim : str):
        self.width : float = 1.0
        self.vertex_coords : numpy.typing.NDArray = np.empty(shape=((0,3) if dim == "3D" else (0,2)), dtype=np.float32)
        self.vertex_colors : numpy.typing.NDArray = np.empty(shape=(0,4), dtype=np.float32)
        self.draw_batch : gpu.types.GPUBatch = None
        self.dirty : bool = False
        self.shader : gpu.types.GPUShader = gpu.shader.from_builtin("SMOOTH_COLOR")

    def create_fresh_batch(self):
        self.draw_batch = gpu_extras.batch.batch_for_shader(self.shader, 'LINES', {"pos" : self.vertex_coords, "color" : self.vertex_colors})
        self.dirty = False

    def draw_callback(self):
        if self.draw_batch == None:
            return
        gpu.state.line_width_set(self.width)
        self.shader.bind()   
        self.draw_batch.draw(self.shader)

class DottedLineRenderData:
    def __init__(self, dim : str):
        self.width : float = 1.0
        self.vertex_coords : numpy.typing.NDArray = np.empty(shape=((0,3) if dim == "3D" else (0,2)), dtype=np.float32)
        self.vertex_colors : numpy.typing.NDArray = np.empty(shape=(0,4), dtype=np.float32)
        self.vertex_arc_lengths : numpy.typing.NDArray = np.empty(shape=(0,1), dtype=np.float32)
        self.draw_batch : gpu.types.GPUBatch = None
        self.dirty : bool = False
        self.shader = shaders.create_dashed_lines_shader()

    def create_fresh_batch(self):
        self.draw_batch = gpu_extras.batch.batch_for_shader(self.shader, 'LINES', {"pos" : self.vertex_coords, 
                                                                                   "arcLength" : self.vertex_arc_lengths, 
                                                                                   "color" : self.vertex_colors})
        self.dirty = False

    def draw_callback(self):
        if self.draw_batch == None:
            return
        gpu.state.line_width_set(self.width)
        self.shader.bind()
        proj_matrix = gpu.matrix.get_projection_matrix()
        if bpy.context.region_data is not None:
            proj_matrix = bpy.context.region_data.perspective_matrix
        self.shader.uniform_float("viewProjectionMatrix", proj_matrix)   
        self.draw_batch.draw(self.shader)

class Canvas:
    def __init__(self, dim : str):
        self.dim = dim
        self.triangles : TriangleRenderData = TriangleRenderData(dim)
        self.lines : LineRenderData = LineRenderData(dim)
        self.dotted_lines : DottedLineRenderData = DottedLineRenderData(dim)

    def draw_callback(self):
        gpu.state.blend_set('ALPHA')
        if (self.dim == "3D"):
            gpu.state.depth_test_set('LESS_EQUAL')
        self.triangles.draw_callback()
        self.dotted_lines.draw_callback()
        self.lines.draw_callback()

def canvas_collection_draw_callback(collection : typing.Iterable[Canvas]):
    for canvas in collection:
        canvas.draw_callback()

class Artist:
    """Interface for drawing stuff to Blenders space view 3d"""
    def __init__(self, context : bpy.types.Context):
        self.context : bpy.types.Context = context
        self.drawing_handle_3d = None # POST VIEW
        self.drawing_handle_2d = None # POST PIXEL
        self.canvas_collection_3d : dict[str, Canvas] = {}
        self.canvas_collection_2d : dict[str, Canvas] = {}

    def hide_all_drawings(self):
        deactivate_draw_callback(self.drawing_handle_3d)
        deactivate_draw_callback(self.drawing_handle_2d)
        self.drawing_handle_3d = None
        self.drawing_handle_2d = None
        redraw_view_3d(self.context)

    def redraw_all(self):
        self.hide_all_drawings()
        for canvas in chain(self.canvas_collection_3d.values(), self.canvas_collection_2d.values()):
            if canvas.triangles.dirty:
                canvas.triangles.create_fresh_batch()
            if canvas.lines.dirty:
                canvas.lines.create_fresh_batch()
            if canvas.dotted_lines.dirty:
                canvas.dotted_lines.create_fresh_batch()
        
        self.drawing_handle_2d = bpy.types.SpaceView3D.draw_handler_add(canvas_collection_draw_callback, (self.canvas_collection_2d.values(),), "WINDOW", "POST_PIXEL")
        self.drawing_handle_3d = bpy.types.SpaceView3D.draw_handler_add(canvas_collection_draw_callback, (self.canvas_collection_3d.values(),), "WINDOW", "POST_VIEW")
        redraw_view_3d(self.context)

    def __del__(self):
        self.hide_all_drawings()

    def clear_layer(self, layer_name : str):
        if layer_name in self.canvas_collection_2d.keys():
            del self.canvas_collection_2d[layer_name]
        if layer_name in self.canvas_collection_3d.keys():
            del self.canvas_collection_3d[layer_name]
    
    def __get_canvas(self, dim : str, layer_name : str):
        if dim == "2D":
            canvas = self.canvas_collection_2d.setdefault(layer_name, Canvas(dim))
        elif dim == "3D":
            canvas = self.canvas_collection_3d.setdefault(layer_name, Canvas(dim))
        return canvas

    def __set_layer_line_width_nd(self, dim : str, layer_name : str, line_width : float):
        canvas = self.__get_canvas(dim, layer_name)
        canvas.lines.width = line_width

    def __set_layer_dotted_line_width_nd(self, dim : str, layer_name : str, line_width : float):
        canvas = self.__get_canvas(dim, layer_name)
        canvas.dotted_lines.width = line_width

    def set_layer_line_width_2d(self, layer_name : str, line_width : float):
        self.__set_layer_line_width_nd("2D", layer_name, line_width)

    def set_layer_line_width_3d(self, layer_name : str, line_width : float):
        self.__set_layer_line_width_nd("3D", layer_name, line_width)

    def set_layer_dotted_line_width_2d(self, layer_name : str, line_width : float):
        self.__set_layer_dotted_line_width_nd("2D", layer_name, line_width)

    def set_layer_dotted_line_width_3d(self, layer_name : str, line_width : float):
        self.__set_layer_dotted_line_width_nd("3D", layer_name, line_width)

    def __draw_triangles_nd(self, dim : str, layer_name : str, v_coords : np.typing.ArrayLike, v_colors : np.typing.ArrayLike, t_indices : np.typing.ArrayLike):
        canvas = self.__get_canvas(dim, layer_name)
        first_vertex_index = canvas.triangles.vertex_coords.shape[0]
        canvas.triangles.vertex_coords = np.concatenate((canvas.triangles.vertex_coords, v_coords), axis=0)
        canvas.triangles.vertex_colors = np.concatenate((canvas.triangles.vertex_colors, v_colors), axis=0)
        canvas.triangles.triangle_indices = np.concatenate((canvas.triangles.triangle_indices, t_indices + first_vertex_index), axis=0)
        canvas.triangles.dirty = True

    def draw_triangles_2d(self, layer_name : str, v_coords : np.typing.ArrayLike, v_colors : np.typing.ArrayLike, t_indices : np.typing.ArrayLike):
        self.__draw_triangles_nd("2D", layer_name, v_coords, v_colors, t_indices)

    def draw_triangles_3d(self, layer_name : str, v_coords : np.typing.ArrayLike, v_colors : np.typing.ArrayLike, t_indices : np.typing.ArrayLike):
        self.__draw_triangles_nd("3D", layer_name, v_coords, v_colors, t_indices)

    def __draw_mesh_triangles_nd(self, dim : str, layer_name : str, mesh : bpy.types.Mesh, face_ids : list[int], colors : np.typing.NDArray, normal_offset : float = 0):
        vertex_coords = []
        vertex_colors = []
        triangle_indices = []
        i = 0
        for face_id, color in zip(face_ids, colors):
            curr_face = mesh.polygons[face_id]
            normal = curr_face.normal
            for v_id in curr_face.vertices:
                vertex_coords.append(mesh.vertices[v_id].co + normal_offset * normal)
                vertex_colors.append(color)
            triangle_indices.append((i, i+1, i+2))
            i += 3
        self.__draw_triangles_nd(dim, layer_name, 
                                 np.array(vertex_coords, np.float32), 
                                 np.array(vertex_colors, dtype=np.float32), 
                                 np.array(triangle_indices, dtype=np.int32))
        
    def draw_mesh_triangles_2d(self, layer_name : str, mesh : bpy.types.Mesh, face_ids : list[int], colors : np.typing.NDArray, normal_offset : float = 0):
        self.__draw_mesh_triangles_nd("2D", layer_name, mesh, face_ids, colors, normal_offset)

    def draw_mesh_triangles_3d(self, layer_name : str, mesh : bpy.types.Mesh, face_ids : list[int], colors : np.typing.NDArray, normal_offset : float = 0):
        self.__draw_mesh_triangles_nd("3D", layer_name, mesh, face_ids, colors, normal_offset)

    def __draw_edges_nd(self, dim : str, layer_name : str, vertex_coords : np.typing.NDArray, vertex_colors : np.typing.NDArray):
        canvas = self.__get_canvas(dim, layer_name)
        canvas.lines.vertex_coords = np.concatenate((canvas.lines.vertex_coords, vertex_coords), axis=0)
        canvas.lines.vertex_colors = np.concatenate((canvas.lines.vertex_colors, vertex_colors), axis=0)
        canvas.lines.dirty = True

    def draw_edges_2d(self, layer_name : str, vertex_coords : np.typing.NDArray, vertex_colors : np.typing.NDArray):
        self.__draw_edges_nd("2D", layer_name, vertex_coords, vertex_colors)

    def draw_edges_3d(self, layer_name : str, vertex_coords : np.typing.NDArray, vertex_colors : np.typing.NDArray):
        self.__draw_edges_nd("3D", layer_name, vertex_coords, vertex_colors)

    def __draw_dotted_edges_nd(self, 
                               dim : str, 
                               layer_name : str, 
                               vertex_coords : np.typing.NDArray, 
                               vertex_colors : np.typing.NDArray, 
                               vertex_arc_lengths : np.typing.NDArray = None):
        canvas = self.__get_canvas(dim, layer_name)
        canvas.dotted_lines.vertex_coords = np.concatenate((canvas.lines.vertex_coords, vertex_coords), axis=0)
        canvas.dotted_lines.vertex_colors = np.concatenate((canvas.lines.vertex_colors, vertex_colors), axis=0)

        if vertex_arc_lengths == None:
            # compute distances between consecutive vertices
            repeated_start_points =  np.repeat(vertex_coords[range(0, vertex_coords.shape[0], 2)], repeats=2, axis=0)
            vertex_arc_lengths = np.linalg.norm(vertex_coords - repeated_start_points, axis=1)

        canvas.dotted_lines.vertex_arc_lengths = np.concatenate((canvas.dotted_lines.vertex_arc_lengths, vertex_arc_lengths), axis=0)
        canvas.dotted_lines.dirty = True

    def draw_dotted_edges_2d(self, 
                             layer_name : str, 
                             vertex_coords : np.typing.NDArray, 
                             vertex_colors : np.typing.NDArray, 
                             vertex_arc_lengths : np.typing.NDArray = None):
        self.__draw_dotted_edges_nd("2D", layer_name, vertex_coords, vertex_colors, vertex_arc_lengths)

    def draw_dotted_edges_3d(self, 
                             layer_name : str, 
                             vertex_coords : np.typing.NDArray, 
                             vertex_colors : np.typing.NDArray,
                             vertex_arc_lengths : np.typing.NDArray = None):
        self.__draw_dotted_edges_nd("3D", layer_name, vertex_coords, vertex_colors, vertex_arc_lengths)
           