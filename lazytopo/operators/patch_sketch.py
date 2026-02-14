import numpy as np
from enum import Enum
import time

# blender api
import bpy
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader
from bpy.props import FloatProperty, IntProperty
from bpy_extras import view3d_utils
from mathutils import Vector

# agplib
import agplib

from . import all_operators
from .operator_helpers import active_object_is_mesh, currently_in_object_mode, is_manifold_trimesh
from ..utils.ui_helpers import trigger_lazytopo_panels_redraw
from ..rendering.rendering_helpers import uniform_lines_2D_draw_callback, redraw_view_3d
from ..rendering.artist import Artist
from ..rendering import colors
from ..patchgraph import SketchedEdgePoint
from ..io.cppadapter import get_agp_mesh_from_bpy_mesh

class EdgeSketchingState(Enum):
    DEFAULT_STATE = 0
    SKETCHING_EDGE = 1
    LOOP_PREVIEW = 2

class SketchPointAddResult(Enum):
    SUCCESS = 0
    NOT_ON_MESH = 1
    TO_CLOSE_TO_LAST = 2

class LAZYTOPO_OT_cancel_edge_sketching(bpy.types.Operator):
    """ Exit the edge sketching mode """
    bl_label = "Exit Edge Sketching"
    bl_idname = "lazytopo.cancel_edge_sketching"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event, message="All sketched edges will be lost!", confirm_text="Ok")

    def execute(self, context):
        if context.window_manager.layzytopo_in_edge_sketch_mode:
            LAZYTOPO_OT_edge_sketching._exit_on_next_event = True
        return {'FINISHED'}
all_operators.append(LAZYTOPO_OT_cancel_edge_sketching)

class LAZYTOPO_OT_edge_sketching(bpy.types.Operator):
    """ Sketch patch edges """
    bl_label = "Sketch Edges"
    bl_idname  = "lazytopo.edge_sketching"

    _exit_on_next_event = False
    DRAW_LAYER_SKETCHING = "sketching_layer"
    DRAW_LAYER_DEBUG = "debug_layer"

    sketch_sample_distance: FloatProperty(
        name="Sample distance",
        description="Distance between two consecutive points on a sketched edge",
        default=0.05,
        min=0.0
    )

    def invoke(self, context, event):
        self.active_object = context.active_object
        self.mesh = context.active_object.data

        bm = bmesh.new()
        bm.from_mesh(self.mesh)
        if not is_manifold_trimesh(bm):
            self.report({'ERROR_INVALID_INPUT'}, "Sketching is only supported for manifold triangle meshes.")
            return {'CANCELLED'}
        bm.free()

        self.cpp_trimesh = get_agp_mesh_from_bpy_mesh(self.mesh)
        self.sketched_edges: list[list[SketchedEdgePoint]] = []
        self.currently_sketched_edge: list[SketchedEdgePoint] = []
        self.editing_state = EdgeSketchingState.DEFAULT_STATE
        context.window_manager.layzytopo_in_edge_sketch_mode = True
        LAZYTOPO_OT_edge_sketching._exit_on_next_event = False
        self.artist = Artist(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def draw_currently_sketched_line(self):
        line_points = []
        for i in range(1, len(self.currently_sketched_edge)):
            line_points.append(self.currently_sketched_edge[i-1].pos_2d)
            line_points.append(self.currently_sketched_edge[i].pos_2d)

        self.artist.clear_layer(self.DRAW_LAYER_SKETCHING)
        self.artist.set_layer_line_width_2d(self.DRAW_LAYER_SKETCHING, 2.0)
        self.artist.draw_edges_2d(self.DRAW_LAYER_SKETCHING,
                                  np.array(line_points, dtype=np.float32), 
                                  np.array([colors.ORANGE for _ in range(len(line_points))], dtype=np.float32))
        self.artist.redraw_all()

    def get_mouse_image_coords(self, context: bpy.types.Context, event :bpy.types.Event):
        mouse_x = event.mouse_region_x
        mouse_y = event.mouse_region_y
        return context.region.view2d.region_to_view(mouse_x, mouse_y)

    def check_if_mouse_is_inside_3d_view(self, context: bpy.types.Context, event):
        mouse_x, mouse_y = event.mouse_x, event.mouse_y
        region_x = context.region.x
        region_y = context.region.y
        if mouse_x < region_x or mouse_x > region_x + context.region.width or mouse_y < region_y or mouse_y > region_y + context.region.height:
            return False
        return True

    def try_add_point_to_current_sketched_edge(self, context: bpy.types.Context, event: bpy.types.Event, ignore_sampling_distance = False) -> SketchPointAddResult:
        region = context.region
        rv3d = context.space_data.region_3d
        coord = (event.mouse_region_x, event.mouse_region_y)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        mesh_hit, location, _, face_index = self.active_object.ray_cast(ray_origin, ray_direction)
        if not mesh_hit:
            return SketchPointAddResult.NOT_ON_MESH
        if not ignore_sampling_distance and len(self.currently_sketched_edge) > 0:
            prev_point: Vector = self.currently_sketched_edge[-1].pos_3d
            if (location - prev_point).length < self.sketch_sample_distance:
                return SketchPointAddResult.TO_CLOSE_TO_LAST
        self.currently_sketched_edge.append(SketchedEdgePoint(pos_2d=coord, pos_3d=location, face_index=face_index))
        return SketchPointAddResult.SUCCESS

    def add_sketched_path_to_patch_graph(self):
        self.sketched_edges.append(self.currently_sketched_edge)

        # test agplib
        try:
            disc_on_mesh = agplib.grow_face_subset_to_disc([e.face_index for e in self.currently_sketched_edge], self.cpp_trimesh, 5)
            
            if disc_on_mesh is not None:
                n_faces = len(disc_on_mesh.face_set.faces)
                self.artist.clear_layer(self.DRAW_LAYER_DEBUG)
                self.artist.draw_mesh_triangles_3d(self.DRAW_LAYER_DEBUG, self.mesh, disc_on_mesh.face_set.faces, [(*colors.MAGENTA[:3], 0.5) for _ in range(n_faces)], 0.001)
                self.artist.redraw_all()
                print(f"Collected {len(disc_on_mesh.face_set.faces)} around sketched edge")
            else:
                print("Faces around edge don't form a disc!")

        except RuntimeError as err:
            print("OH NO")
            print(err)
        except Exception as exc:
            print("OH NONONO")
            print(exc)            

    def exit_modal_mode(self, context):
        context.window_manager.layzytopo_in_edge_sketch_mode = False
        self.artist.hide_all_drawings()
        trigger_lazytopo_panels_redraw(context)
        return {"FINISHED"} 

    def modal(self, context, event : bpy.types.Event):
        if LAZYTOPO_OT_edge_sketching._exit_on_next_event:
            return self.exit_modal_mode(context)
        if context.region is None or context.region.view2d is None:
            return self.exit_modal_mode(context)
        
        match self.editing_state:
            case EdgeSketchingState.DEFAULT_STATE:
                if event.type == "ESC":
                    bpy.ops.lazytopo.cancel_edge_sketching('INVOKE_DEFAULT')
                    return {'RUNNING_MODAL'}
                if event.type == "RET":
                    print("Applying sketched edges...")
                    return self.exit_modal_mode(context)
                if event.type == "LEFTMOUSE" and event.value == "PRESS":
                    if self.try_add_point_to_current_sketched_edge(context, event, ignore_sampling_distance=True) == SketchPointAddResult.SUCCESS:
                        self.editing_state = EdgeSketchingState.SKETCHING_EDGE
                        return {'RUNNING_MODAL'}
            case EdgeSketchingState.SKETCHING_EDGE:
                if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                    if len(self.currently_sketched_edge) > 0:
                        self.add_sketched_path_to_patch_graph()
                    self.artist.clear_layer(self.DRAW_LAYER_SKETCHING)
                    self.artist.redraw_all()
                    self.currently_sketched_edge = []
                    self.editing_state = EdgeSketchingState.DEFAULT_STATE
                if event.type == "MOUSEMOVE":
                    sketch_result = self.try_add_point_to_current_sketched_edge(context, event)
                    if sketch_result == SketchPointAddResult.NOT_ON_MESH:
                        self.artist.clear_layer(self.DRAW_LAYER_SKETCHING)
                        self.artist.redraw_all()
                        self.currently_sketched_edge = []
                        self.editing_state = EdgeSketchingState.DEFAULT_STATE
                    if sketch_result == SketchPointAddResult.SUCCESS:
                        self.draw_currently_sketched_line()
                return {'RUNNING_MODAL'}
            case EdgeSketchingState.LOOP_PREVIEW:
                pass

        return {'PASS_THROUGH'}

    def __del__(self):
        # just to be super safe here
        self.artist.hide_all_drawings()
        bpy.context.window_manager.layzytopo_in_edge_sketch_mode = False

    @classmethod
    def poll(cls, context):
        return active_object_is_mesh(context) and currently_in_object_mode(context)
all_operators.append(LAZYTOPO_OT_edge_sketching)
