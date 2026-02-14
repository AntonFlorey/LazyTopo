import numpy as np
import time

# blender api
import bpy
from bpy.props import FloatProperty, IntProperty

# agplib
import agplib

# addon
from . import all_operators
from .operator_helpers import active_object_is_mesh
from ..utils import ui_helpers, constants, contexts
from ..io import attribute_helpers
from ..rendering.crossfield_visualization import update_all_crosses

class LAZYTOPO_OT_compute_crossfield(bpy.types.Operator):
    """ Compute a crossfield that guides sketching quad placement """
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
        description="Decides how much the cross field computation gets guided by principal curvature direction",
        default=1.0,
        min=0.0
    )

    def draw(self, context):
        layout = self.layout
        split_factor = 0.7
        ui_helpers.write_custom_split_property_row(layout, "Optimization Rounds", self.properties, "max_iters", split_factor)
        ui_helpers.write_custom_split_property_row(layout, "Max Layers", self.properties, "max_layers", split_factor)
        ui_helpers.write_custom_split_property_row(layout, "Curvature weight", self.properties, "principal_curvature_weight", split_factor)

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self, title="Crossfield Options")

    def execute(self, context):
        with contexts.read_mesh_and_bmesh_in_object_mode(context) as (mesh, bm):
            crossfield_settings = context.scene.lazytopo_crossfield_settings

            surface_graph = agplib.crossfield.SurfaceGraph()
            for face in bm.faces:
                node_index = surface_graph.add_node(agplib.crossfield.SurfaceGraphNode(face.calc_area(), np.array(face.normal)))
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
                vertices_with_normals = [agplib.crossfield.VertexWithNormal(v.co, v.normal) for v in face.verts]
                principal_curvature_info = agplib.crossfield.compute_principal_curvature(vertices_with_normals, face.normal)
                principal_curvature_directions.append(principal_curvature_info.direction)
                principal_curvature_unambiguities.append(principal_curvature_info.unambiguity)
                if self.principal_curvature_weight == 0 or principal_curvature_info.unambiguity < crossfield_settings.curvature_threshold:
                    continue
                curvature_constraints.append(agplib.crossfield.CrossConstraint(self.principal_curvature_weight * principal_curvature_info.unambiguity, principal_curvature_info.direction, face.index))
            
            print("Computing crossfield...")
            star_time = time.time()
            crossfield = np.asarray(agplib.crossfield.compute_crossfield(surface_graph, curvature_constraints, max_iters=self.max_iters, max_multires_layers=self.max_layers), dtype=np.float64)
            print("Done after", time.time() - star_time, "seconds.")

            # Save the principal curvature info
            curvature_attribute = attribute_helpers.create_new_or_overwrite_attribute(mesh.attributes, constants.PRINCIPAL_CURVATURE_ATTR_NAME, "FLOAT_VECTOR", "FACE")
            curvature_attribute.data.foreach_set("vector", np.asarray(principal_curvature_directions).flatten())
            curvature_unambiguity_attribute = attribute_helpers.create_new_or_overwrite_attribute(mesh.attributes, constants.PRINCIPAL_CURVATURE_UNAMBIGUITY_ATTR_NAME, "FLOAT", "FACE")
            curvature_unambiguity_attribute.data.foreach_set("value", principal_curvature_unambiguities)

            # Save the crossfield
            crossfield_attribute = attribute_helpers.create_new_or_overwrite_attribute(mesh.attributes, constants.CROSSFIELD_ATTR_NAME, "FLOAT_VECTOR", "FACE")
            crossfield_attribute.data.foreach_set("vector", crossfield.flatten())

            update_all_crosses(self, context)

        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        return active_object_is_mesh(context)
all_operators.append(LAZYTOPO_OT_compute_crossfield)
