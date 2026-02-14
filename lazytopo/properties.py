import bpy
from bpy.types import Scene

from bpy.props import BoolProperty
# from bpy.props import CollectionProperty
# from bpy.props import EnumProperty
from bpy.props import FloatProperty
# from bpy.props import IntProperty
# from bpy.props import PointerProperty
# from bpy.props import StringProperty
# from bpy.props import PropertyGroup

from .rendering import crossfield_visualization

class CrossfieldSettings(bpy.types.PropertyGroup):
    show_crossfield: BoolProperty(
        name="Show Crossfield",
        default=True,
        update=crossfield_visualization.update_all_crosses
    )
    show_minmax_curvature: BoolProperty(
        name="Show MinMax Curvature",
        default=True,
        update=crossfield_visualization.update_all_crosses
    )
    curvature_threshold: FloatProperty(
        name="Curvature threshold",
        default=0.0,
        min=0.0,
        update=crossfield_visualization.update_all_crosses  
    )

# This is where you assign any variables you need in your script. Note that they
# won't always be assigned to the Scene object but it's a good place to start.
def register():
    bpy.utils.register_class(CrossfieldSettings)
    Scene.lazytopo_crossfield_settings = bpy.props.PointerProperty(type=CrossfieldSettings)
    bpy.types.WindowManager.layzytopo_in_edge_sketch_mode = BoolProperty(name="Sketching edges with Lazytopo", default=False)

def unregister():
    bpy.utils.unregister_class(CrossfieldSettings)
    del Scene.lazytopo_crossfield_settings
    bpy.types.WindowManager.lazytopo_in_edge_sketch_mode = False
    del bpy.types.WindowManager.layzytopo_in_edge_sketch_mode
    