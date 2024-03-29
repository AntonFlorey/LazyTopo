import bpy
import bmesh
from bpy.types import Context
from .crossfield import CrossField, MultiResCrossField
from .drawing import show_crossfield, hide_crossfield, show_crossfield_graph, hide_crossfield_graph, show_hierarchy, hide_hierarchy


class TestOperator(bpy.types.Operator):
    bl_label = "Test Operator"
    bl_idname  = "wm.test_op"
    _handle = None

    my_float : bpy.props.FloatProperty(name="TestFloat")
    
    def execute(self, context):
        print("This is a simple test")
        print("User input float: ", self.my_float)
        hide_crossfield()
        hide_crossfield_graph()
        hide_hierarchy()
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        # get the currently selected object
        ao = bpy.context.active_object
        self.my_mr_crossfield = MultiResCrossField(ao)
        self.my_mr_crossfield.optimize()
        #self.my_crossfield = CrossField(ao)

        topo_settings = context.scene.lazytopo_settings
        if topo_settings.my_bool:
            show_crossfield(self.my_mr_crossfield.cross_points_for_rendering(level=topo_settings.crossfield_level_shown))
        
        if topo_settings.show_crossfield_graph:
            show_crossfield_graph(self.my_mr_crossfield.graph_points_for_rendering(level=topo_settings.crossfield_level_shown))

        if topo_settings.color_crossfield_hierarchy:
            show_hierarchy(*self.my_mr_crossfield.merged_faces_for_rendering(level=topo_settings.crossfield_level_shown))

        return wm.invoke_props_dialog(self)

    def draw(self, context: Context):
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.prop(self, "my_float")
        row = col.row()
        row.operator("wm.sub_op")

    def cancel(self, context: Context):
        hide_crossfield()
        hide_crossfield_graph()
        hide_hierarchy()

class SubOperator(bpy.types.Operator):
    bl_label = "Sub-Operator"
    bl_idname  = "wm.sub_op"

    def execute(self, context):
        print("I am a nested operator")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context: Context):
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.label(text="I am a nested  operator")


def register():
    bpy.utils.register_class(TestOperator)
    bpy.utils.register_class(SubOperator)

def unregister():
    bpy.utils.unregister_class(TestOperator)
    bpy.utils.unregister_class(SubOperator)
