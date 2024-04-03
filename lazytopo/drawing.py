import bpy
import gpu
import bgl
from gpu_extras.batch import batch_for_shader
from . import crossfield
from . import topo_globals

# Keep track of active draw callbacks 
_drawing_handle_crossfield = None
_drawing_handle_constraints = None
_drawing_handle_crossfield_graph = None
_drawing_handle_crossfield_hierarchy = None
_drawing_handle_singularities = None
_drawing_handle_debug = None

class ColorGenerator():
    """ A simple color generator"""

    def __init__(self):
        self.colors = {
            "BLUE" :        (  0 / 255,  84 / 255, 159 / 255, 1),
            "MAGENTA" :     (227 / 255,   0 / 255, 102 / 255, 1),
            "YELLOW" :      (255 / 255, 237 / 255,   0 / 255, 1),
            "PETROL" :      (  0 / 255,  97 / 255, 101 / 255, 1),
            "TEAL" :        (  0 / 255, 152 / 255, 161 / 255, 1),
            "GREEN" :       ( 87 / 255, 171 / 255,  39 / 255, 1),
            "MAY_GREEN" :   (189 / 255, 205 / 255,   0 / 255, 1),
            "ORANGE" :      (246 / 255, 168 / 255,   0 / 255, 1),
            "RED" :         (204 / 255,   7 / 255,  30 / 255, 1),
            "BORDEAUX" :    (161 / 255,  16 / 255,  53 / 255, 1),
            "PURPLE" :      ( 97 / 255,  33 / 255,  88 / 255, 1),
            "LILAC" :       (122 / 255, 111 / 255, 172 / 255, 1)
        }
        self.index = 0

    def next_color(self):
        col = list(self.colors.values())[self.index]
        self.index = (self.index + 1) % len(self.colors)
        return col

def update_all_crossfield_drawings(self, context):
    try:
        settings = context.scene.lazytopo_settings
    except AttributeError:
        print("Context not yet available...")
        return
    # hide everything
    hide_crossfield()
    hide_constraints()
    hide_crossfield_graph()
    hide_hierarchy()
    hide_singularities()
    hide_debug()

    if topo_globals.active_crossfield is not None:
        if settings.show_crossfield:
            show_crossfield(topo_globals.active_crossfield.cross_points_for_rendering(settings.crossfield_level_shown))
        if settings.show_constraints:
            show_constraints(topo_globals.active_crossfield.constraint_crosses_for_rendering(settings.crossfield_level_shown))
        if settings.show_crossfield_graph:
            show_crossfield_graph(topo_globals.active_crossfield.graph_points_for_rendering(settings.crossfield_level_shown))
        if settings.show_singularities:
            show_singularities(topo_globals.active_crossfield.singularity_points_for_rendering())
        if settings.color_crossfield_hierarchy:
            show_hierarchy(*topo_globals.active_crossfield.merged_faces_for_rendering(settings.crossfield_level_shown))

def hide_crossfield():
    global _drawing_handle_crossfield
    if _drawing_handle_crossfield is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_drawing_handle_crossfield, 'WINDOW')
        _drawing_handle_crossfield = None

def show_crossfield(crossfield_as_line_array):
    global _drawing_handle_crossfield
    hide_crossfield() # remove old drawing 
    _drawing_handle_crossfield = bpy.types.SpaceView3D.draw_handler_add(crossfield_draw_callback, (crossfield_as_line_array,), "WINDOW", "POST_VIEW")

def crossfield_draw_callback(crosses_as_lines):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.line_width_set(2.0)
    batch = batch_for_shader(shader, 'LINES', {"pos": crosses_as_lines})
    shader.uniform_float("color", (0.0, 0.0, 1.0, 1.0))

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    batch.draw(shader)

    # restore opengl defaults
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(1.0)

def hide_constraints():
    global _drawing_handle_constraints
    if _drawing_handle_constraints is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_drawing_handle_constraints, 'WINDOW')
        _drawing_handle_constraints = None

def show_constraints(constraints_as_line_array):
    global _drawing_handle_constraints
    hide_constraints() # remove old drawing 
    _drawing_handle_constraints = bpy.types.SpaceView3D.draw_handler_add(constraints_draw_callback, (constraints_as_line_array,), "WINDOW", "POST_VIEW")

def constraints_draw_callback(constraints_as_line_array):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.line_width_set(2.0)
    batch = batch_for_shader(shader, 'LINES', {"pos": constraints_as_line_array})
    shader.uniform_float("color", (204 / 255,   7 / 255,  30 / 255, 1))

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    batch.draw(shader)

    # restore opengl defaults
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(1.0)

def hide_crossfield_graph():
    global _drawing_handle_crossfield_graph
    if _drawing_handle_crossfield_graph is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_drawing_handle_crossfield_graph, 'WINDOW')
        _drawing_handle_crossfield_graph = None

def show_crossfield_graph(crossfield_graph_as_line_array):
    global _drawing_handle_crossfield_graph
    hide_crossfield_graph() # remove old drawing 
    _drawing_handle_crossfield_graph = bpy.types.SpaceView3D.draw_handler_add(crossfield_graph_draw_callback, (crossfield_graph_as_line_array,), "WINDOW", "POST_VIEW")

def crossfield_graph_draw_callback(crossfield_graph_as_line_array):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.line_width_set(2.0)
    batch = batch_for_shader(shader, 'LINES', {"pos": crossfield_graph_as_line_array})
    shader.uniform_float("color", (0.0, 1.0, 0, 1.0))

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    batch.draw(shader)

    # restore opengl defaults
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(1.0)

def hide_hierarchy():
    global _drawing_handle_crossfield_hierarchy
    if _drawing_handle_crossfield_hierarchy is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_drawing_handle_crossfield_hierarchy, 'WINDOW')
        _drawing_handle_crossfield_hierarchy = None

def show_hierarchy(vertex_positions, triangle_indices):
    global _drawing_handle_crossfield_hierarchy
    hide_hierarchy() # remove old drawing 
    _drawing_handle_crossfield_hierarchy = bpy.types.SpaceView3D.draw_handler_add(hierarchy_draw_callback, (vertex_positions, triangle_indices), "WINDOW", "POST_VIEW")

def hierarchy_draw_callback(vertex_positions, triangle_indices):
    shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
    gpu.state.face_culling_set("BACK")
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)

    color_generator = ColorGenerator()
    for curr_indices in triangle_indices:
        batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions}, indices=curr_indices)
        shader.uniform_float("color", color_generator.next_color())
        batch.draw(shader)

    # restore opengl defaults
    gpu.state.depth_mask_set(False)

def hide_singularities():
    global _drawing_handle_singularities
    if _drawing_handle_singularities is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_drawing_handle_singularities, 'WINDOW')
        _drawing_handle_singularities = None

def show_singularities(singularities):
    global _drawing_handle_singularities
    hide_singularities() # remove old drawing 
    _drawing_handle_singularities = bpy.types.SpaceView3D.draw_handler_add(singularities_draw_callback, (singularities,), "WINDOW", "POST_VIEW")

def singularities_draw_callback(singularities):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
 
    color_generator = ColorGenerator()
    for index, singularities_pos in singularities.items():
        batch = batch_for_shader(shader, "POINTS", {"pos" : singularities_pos})
        shader.uniform_float("color", color_generator.next_color())
        batch.draw(shader)

    # restore opengl defaults
    gpu.state.depth_mask_set(False)

def hide_debug():
    global _drawing_handle_debug
    if _drawing_handle_debug is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_drawing_handle_debug, 'WINDOW')
        _drawing_handle_debug = None

def show_debug(debug_points):
    global _drawing_handle_debug
    hide_debug() # remove old drawing 
    _drawing_handle_debug = bpy.types.SpaceView3D.draw_handler_add(debug_point_draw_callback, (debug_points,), "WINDOW", "POST_VIEW")

def debug_point_draw_callback(debug_points):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    
    batch = batch_for_shader(shader, "POINTS", {"pos" : debug_points})
    shader.uniform_float("color", (204 / 255,   7 / 255,  30 / 255, 1))
    batch.draw(shader)

    # restore opengl defaults
    gpu.state.depth_mask_set(False)