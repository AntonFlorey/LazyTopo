import bpy
from bpy.types import Panel

from .properties import RenderingSettings

class LAZYTOPO_PT_main_panel(Panel):
    bl_label = "LazyTopo"
    bl_idname = "LAZYTOPO_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LazyTopo"

    def draw(self, context: bpy.types.Context):
        rendering_props : RenderingSettings = context.scene.lazytopo_settings
        layout = self.layout
        row = layout.row()
        row.label(text="LazyTopo Addon", icon="MESH_CUBE")
        row = layout.row()
        row.operator("lazytopo.compute_crossfield", text="Compute Crossfield", icon="MOD_WIREFRAME")
        row = layout.row()
        row.prop(rendering_props, "show_crossfield")
        row = layout.row()
        row.prop(rendering_props, "curvature_threshold")

def register():
    bpy.utils.register_class(LAZYTOPO_PT_main_panel)

def unregister():
    bpy.utils.unregister_class(LAZYTOPO_PT_main_panel)
