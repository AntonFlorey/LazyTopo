import numpy as np

# blender api
import bpy
import bmesh
from bpy.props import BoolProperty, FloatProperty, IntProperty

# agplib
import agplib

# addon
from .utils import ui_helpers, constants
from .io import attribute_helpers

class LAZYTOPO_OT_compute_crossfield(bpy.types.Operator):
    bl_label = "Compute Crossfield"
    bl_idname  = "lazytopo.compute_crossfield"

    max_iters : IntProperty(
        name="Optimization Rounds",
        default=100,
        min=0
    )
    max_layers : IntProperty(
        name="Max Multires Layers",
        default=10,
        min=0
    )
    principal_curvature_weight: FloatProperty(
        name="Curvature weight",
        description="Decides how much the cross field computation gets guided by principal curvature direction.",
        default=1.0,
        min=0.0
    )
    min_principal_curvature_unambiguity: FloatProperty(
        name="Min curvature unambiguity",
        description="All principal curvature directions with an unambuiguity value below this value are ignored.",
        default=0.0,
        min=0.0
    )

    def draw(self, context):
        layout = self.layout
        split_factor = 0.7
        ui_helpers.write_custom_split_property_row(layout, "Optimization Rounds", self.properties, "max_iters", split_factor)
        ui_helpers.write_custom_split_property_row(layout, "Max Layers", self.properties, "max_layers", split_factor)
        ui_helpers.write_custom_split_property_row(layout, "Curvature weight", self.properties, "principal_curvature_weight", split_factor)
        ui_helpers.write_custom_split_property_row(layout, "Min curvature unambiguity", self.properties, "min_principal_curvature_unambiguity", split_factor)

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self, title="Crossfield Options")

    def execute(self, context):
        # get the currently selected object
        ao = bpy.context.active_object
        mesh = ao.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        surface_graph = agplib.SurfaceGraph()
        for face in bm.faces:
            node_index = surface_graph.add_node(agplib.SurfaceGraphNode(face.calc_area(), np.array(face.normal)))
            assert node_index == face.index

        for edge in bm.edges:
            if len(edge.link_faces) != 2:
                continue
            surface_graph.add_edge(edge.link_faces[0].index, edge.link_faces[1].index)

        # Compute principal curvatures for each face
        curvature_constraints = []
        principal_curvature_directions = []
        principal_curvature_unambiguities = []
        for face in bm.faces:
            vertices_with_normals = [agplib.VertexWithNormal(v.co, v.normal) for v in face.verts]
            principal_curvature_info = agplib.compute_principal_curvature(vertices_with_normals, face.normal)
            principal_curvature_directions.append(principal_curvature_info.direction)
            principal_curvature_unambiguities.append(principal_curvature_info.unambiguity)
            if self.principal_curvature_weight == 0 or principal_curvature_info.unambiguity < self.min_principal_curvature_unambiguity:
                continue
            curvature_constraints.append(agplib.CrossConstraint(self.principal_curvature_weight * principal_curvature_info.unambiguity, principal_curvature_info.direction, face.index))
        
        crossfield = np.asarray(agplib.compute_crossfield(surface_graph, curvature_constraints, max_iters=self.max_iters, max_multires_layers=self.max_layers), dtype=np.float64)

        # Save the principal curvature info
        curvature_attribute = attribute_helpers.create_new_or_overwrite_attribute(mesh.attributes, constants.PRINCIPAL_CURVATURE_ATTR_NAME, "FLOAT_VECTOR", "FACE")
        curvature_attribute.data.foreach_set("vector", np.asarray(principal_curvature_directions).flatten())
        curvature_unambiguity_attribute = attribute_helpers.create_new_or_overwrite_attribute(mesh.attributes, constants.PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME, "FLOAT", "FACE")
        curvature_unambiguity_attribute.data.foreach_set("value", principal_curvature_unambiguities)

        # Save the crossfield
        crossfield_attribute = attribute_helpers.create_new_or_overwrite_attribute(mesh.attributes, constants.CROSSFIELD_ATTR_NAME, "FLOAT_VECTOR", "FACE")
        crossfield_attribute.data.foreach_set("vector", crossfield.flatten())

        bm.free()
        return {'FINISHED'}

def register():
    bpy.utils.register_class(LAZYTOPO_OT_compute_crossfield)

def unregister():
    bpy.utils.unregister_class(LAZYTOPO_OT_compute_crossfield)
