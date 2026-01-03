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

class RenderingSettings(bpy.types.PropertyGroup):
    show_crossfield: BoolProperty(
        name="Show Crossfield",
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
    bpy.utils.register_class(RenderingSettings)
    Scene.lazytopo_settings = bpy.props.PointerProperty(type=RenderingSettings)

def unregister():
    bpy.utils.unregister_class(RenderingSettings)
    del Scene.lazytopo_settings
    