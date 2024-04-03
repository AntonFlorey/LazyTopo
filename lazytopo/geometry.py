import numpy as np
import bpy
import bmesh
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion

def any_tangent_basis(normal : Vector):
    """ Returns an arbitrary orthonormal basis of the tangent plane defined by the given normal """

    u = Vector.orthogonal(normal).normalized()
    v = normal.cross(u).normalized()
    return u, v

def estimate_II_fundamental_form(face : bmesh.types.BMFace, u : Vector, v : Vector):
    """ Least squares approximation of the 2nd fundamental form for a given face """

    num_constraints = 2 * len(face.edges)
    A = np.zeros(shape=(num_constraints, 3)) 
    rhs = np.zeros(num_constraints)

    e : bmesh.types.BMEdge = face.edges[0]
    e.verts

    for _i, edge in enumerate(face.edges):
        i = 2 * _i
        v0 : bmesh.types.BMVert = edge.verts[0]
        v1 : bmesh.types.BMVert = edge.verts[1]
        p0 : Vector = v0.co
        p1 : Vector = v1.co
        n0 : Vector = v0.normal
        n1 : Vector = v1.normal
        dv : Vector = p1- p0
        dn : Vector = n1 - n0
        A[i,:] = [dv.dot(u), dv.dot(v), 0]
        A[i+1,:] = [0, dv.dot(u), dv.dot(v)]
        rhs[i] = dn.dot(u)
        rhs[i+1] = dn.dot(v)
    
    x = np.linalg.lstsq(A, rhs, rcond=None)[0]

    # print("lstsq solution:", x)

    II_ff = np.array([[x[0], x[1]], [x[1], x[2]]])

    return II_ff

def principal_curvature_dir_with_score(face : bmesh.types.BMFace):
    """ Returns one tangent vector in the direction of principal curvature together with its significance """

    u, v = any_tangent_basis(face.normal)
    II = estimate_II_fundamental_form(face, u, v)
    
    # Eigen Decomposition
    eigvals, eigvecs = np.linalg.eigh(II)

    # Compute score as the distance between principal curvatures
    direction_unambiguity = abs(eigvals[0] - eigvals[1])

    # Convert the first eigenvector back to 3D in tangent plane
    dir2D = eigvecs[:, 0]
    principal_curvature_dir = (u * dir2D[0] + v * dir2D[1]).normalized()

    return principal_curvature_dir, float(direction_unambiguity)