bl_info = {
    "name": "LazyTopo",
    "author": "Anton Florey",
    "version": (1,0,0),
    "blender": (5,0,0),
    "location": "View3D",
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh"
}

if "bpy" in locals():
    import importlib
    importlib.reload(locals()["ui"])
    importlib.reload(locals()["operators"])
    importlib.reload(locals()["properties"])
else:
    import bpy
    from . import ui
    from . import operators
    from . import properties
    from .rendering import crossfield_visualization

def register():
    # Properties first!
    properties.register()
    # Other stuff later
    operators.register()
    ui.register()

def unregister():
    crossfield_visualization.hide_crossfield()
    operators.unregister()
    ui.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()