import bpy
from bpy.types import Panel

from .properties import CrossfieldSettings

class LAZYTOPO_PT_main_panel(Panel):
    bl_label = "LazyTopo"
    bl_idname = "LAZYTOPO_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LazyTopo"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        row = layout.row()
        row.label(text="thanks for using me!", icon="FUND")

class LAZYTOPO_PT_sketching_panel(bpy.types.Panel):
    bl_label = "Sketching"
    bl_idname = "LAZYTOPO_PT_sketching_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LazyTopo"
    bl_parent_id = "LAZYTOPO_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        
        editing_box = layout.box()
        if not context.window_manager.layzytopo_in_edge_sketch_mode:
            editing_box.row().operator("lazytopo.edge_sketching")
        else:
            editing_box.row().operator("lazytopo.cancel_edge_sketching")
            editing_box.row().label(text="Sketch an edge by holding LMB", icon="MOUSE_LMB")
            editing_box.row().label(text="Sketch loops with shift+LMB", icon="FORCE_MAGNETIC")


class LAZYTOPO_PT_crossfield_panel(bpy.types.Panel):
    bl_label = "Crossfield"
    bl_idname = "LAZYTOPO_PT_crossfield_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LazyTopo"
    bl_parent_id = "LAZYTOPO_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context):
        crossfield_settings : CrossfieldSettings = context.scene.lazytopo_crossfield_settings
        layout = self.layout
        row = layout.row()
        row.operator("lazytopo.compute_crossfield", text="Compute Crossfield", icon="MESH_GRID")
        row = layout.row()
        row.prop(crossfield_settings, "show_crossfield")
        row = layout.row()
        row.prop(crossfield_settings, "show_minmax_curvature")
        row = layout.row()
        row.prop(crossfield_settings, "curvature_threshold")

def register():
    bpy.utils.register_class(LAZYTOPO_PT_main_panel)
    bpy.utils.register_class(LAZYTOPO_PT_sketching_panel)
    bpy.utils.register_class(LAZYTOPO_PT_crossfield_panel)

def unregister():
    bpy.utils.unregister_class(LAZYTOPO_PT_main_panel)
    bpy.utils.unregister_class(LAZYTOPO_PT_sketching_panel)
    bpy.utils.unregister_class(LAZYTOPO_PT_crossfield_panel)
