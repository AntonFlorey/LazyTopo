from bpy.utils import register_classes_factory

all_operators = []

from . import crossfield
register, unregister = register_classes_factory(all_operators)
