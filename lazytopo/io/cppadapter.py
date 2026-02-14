import numpy as np
import agplib
import time

import bpy
from bpy.types import Mesh

def get_agp_mesh_from_bpy_mesh(mesh : Mesh) -> agplib.TriMesh:
    """ Assumes the given mesh is a manifold triangle mesh! """

    mesh_read_time_start = time.time()
    n_verts = len(mesh.vertices)
    n_edges = len(mesh.edges)
    n_faces = len(mesh.polygons)

    # vertex coords
    v_coords = np.zeros(n_verts * 3, np.float64)
    mesh.vertices.foreach_get("co", v_coords)
    v_coords = v_coords.reshape((n_verts, 3))

    # vertex normals
    v_normals = np.zeros(n_verts * 3, np.float64)
    mesh.vertices.foreach_get("normal", v_normals)
    v_normals = v_normals.reshape((n_verts, 3))

    # edge vertices
    e_verts = np.empty(n_edges * 2, np.int32)
    mesh.edges.foreach_get("vertices", e_verts)

    # face normals
    f_normals = np.zeros(n_faces * 3, np.float64)
    mesh.polygons.foreach_get("normal", f_normals)
    f_normals = f_normals.reshape((n_faces, 3))

    # face vertices (has to be triangle mesh!)
    f_verts = np.zeros(n_faces * 3, np.int32)
    mesh.polygons.foreach_get("vertices", f_verts)

    blender_attributes_read_time = time.time()
    print("Reading all Blender mesh attributes took ", blender_attributes_read_time - mesh_read_time_start, "seconds.")

    trimesh = agplib.TriMesh(v_coords, v_normals, e_verts, f_normals, f_verts)
    print("Initializing agplib.Trimesh took ", time.time() - blender_attributes_read_time, "seconds.")

    return trimesh
