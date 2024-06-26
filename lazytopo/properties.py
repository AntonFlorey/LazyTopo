import bpy
from bpy.types import Scene
from .drawing import update_all_crossfield_drawings


# For more information about Blender Properties, visit:
# <https://blender.org/api/blender_python_api_2_78a_release/bpy.types.Property.html>
from bpy.props import BoolProperty
# from bpy.props import CollectionProperty
# from bpy.props import EnumProperty
# from bpy.props import FloatProperty
# from bpy.props import IntProperty
# from bpy.props import PointerProperty
# from bpy.props import StringProperty
# from bpy.props import PropertyGroup

#
# Add additional functions or classes here
#

class MySettings(bpy.types.PropertyGroup):
    show_crossfield: bpy.props.BoolProperty(
        name="Show Crossfield",
        default=True,
        update=update_all_crossfield_drawings
    )

    show_constraints: bpy.props.BoolProperty(
        name="Show Constraints",
        default=True,
        update=update_all_crossfield_drawings
    )

    show_crossfield_graph: bpy.props.BoolProperty(
        name="Show Crossfield Graph",
        default=True,
        update=update_all_crossfield_drawings
    )

    show_singularities: bpy.props.BoolProperty(
        name="Show Crossfield Singularities",
        default=True,
        update=update_all_crossfield_drawings 
    )
    
    color_crossfield_hierarchy : bpy.props.BoolProperty(
        name="Show Crossfield  Hierarchy",
        default=True,
        update=update_all_crossfield_drawings
    )

    crossfield_level_shown: bpy.props.IntProperty(
        name="Crossfield Level shown",
        default=0,
        min=0,
        max=100,
        update=update_all_crossfield_drawings
    )

# This is where you assign any variables you need in your script. Note that they
# won't always be assigned to the Scene object but it's a good place to start.
def register():
    bpy.utils.register_class(MySettings)
    Scene.my_property = BoolProperty(name="Display Stuff",default=True)
    Scene.lazytopo_settings = bpy.props.PointerProperty(type=MySettings)

def unregister():
    bpy.utils.unregister_class(MySettings)
    del Scene.my_property
    del Scene.lazytopo_settings