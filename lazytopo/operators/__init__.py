from bpy.utils import register_classes_factory

all_operators = []

from . import crossfield, patch_sketch
register, unregister = register_classes_factory(all_operators)
