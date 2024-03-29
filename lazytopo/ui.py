import bpy
from bpy.types import Panel

class TestPanel(bpy.types.Panel):
    bl_label = "Test Panel"
    bl_idname  = "LAZYTOPO_PT_TestPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "My first Addon"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        row = layout.row()

        row.label(text="dummy text", icon="FUND")
        layout.operator("wm.test_op")

class SubPanel(bpy.types.Panel):
    bl_label = "I am a child"
    bl_idname  = "LAZYTOPO_PT_PanelB"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "My first Addon"
    bl_parent_id = "LAZYTOPO_PT_TestPanel"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        scene = context.scene
        topo_settings = scene.lazytopo_settings
        row = layout.row()
        row.label(text="I want to draw stuff", icon="GREASEPENCIL")
        row = layout.row()
        row.prop(topo_settings, "my_bool")
        row = layout.row()
        row.prop(topo_settings, "show_singularities")
        row = layout.row()
        row.prop(topo_settings, "show_crossfield_graph")
        row = layout.row()
        row.prop(topo_settings, "color_crossfield_hierarchy")
        row = layout.row()
        row.prop(topo_settings, "crossfield_level_shown")
        

def register():
    bpy.utils.register_class(TestPanel)
    bpy.utils.register_class(SubPanel)

def unregister():
    bpy.utils.unregister_class(TestPanel)
    bpy.utils.unregister_class(SubPanel)
    