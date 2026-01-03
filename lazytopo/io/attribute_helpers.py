import numpy as np

#  blender api
import bpy
from bpy.types import AttributeGroupMesh

def create_new_or_overwrite_attribute(attribute_group : AttributeGroupMesh, attribute_name : str, attribute_type : str, attribute_domain : str):
    if attribute_name in attribute_group:
        attribute_group.remove(attribute_group[attribute_name])
    return attribute_group.new(name=attribute_name, type=attribute_type, domain=attribute_domain)

def read_numpy_array_from_float_attribute(attribute_group : AttributeGroupMesh, attribute_name : str, domain_size :int):
    read_buffer = np.zeros(domain_size, dtype=np.float32)
    attribute_group[attribute_name].data.foreach_get("value", read_buffer)
    return read_buffer

def read_numpy_array_from_vector_attribute(attribute_group : AttributeGroupMesh, attribute_name : str, domain_size :int):
    read_buffer = np.zeros(domain_size * 3, dtype=np.float32)
    attribute_group[attribute_name].data.foreach_get("vector", read_buffer)
    return read_buffer.reshape((domain_size, 3))
