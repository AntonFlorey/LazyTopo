bl_info = {
    "name": "Retopoloco",
    "author": "Anton Florey",
    "version": (1,0),
    "blender": (3,5,1),
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
    from .lazytopo import drawing
    from .lazytopo import ui
    from .lazytopo import operators
    from .lazytopo import properties

def register():
    # Properties first!
    properties.register()
    # Other stuff later
    operators.register()
    ui.register()
    
def unregister():
    operators.unregister()
    ui.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()