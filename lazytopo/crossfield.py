import copy
import bpy
from bpy.types import Context
import bmesh
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion
import bl_math
import time
from .geometry import principal_curvature_dir_with_score


class MultiResCrossField():
    """
    A class that represents a cross field on multiple resolutions of the same mesh
    """
    def __init__(self, ao : bpy.types.Object, max_merge_angle=60, num_layers=10) -> None:
        self.mesh = bmesh.new()
        bpy.ops.object.mode_set(mode="EDIT")
        self.mesh = bmesh.from_edit_mesh(ao.data).copy()
        self.world_matrix = ao.matrix_world
        bpy.ops.object.mode_set(mode="OBJECT")

        # multi res crossfield stored in lists
        self.fields = []
        self.mr_ref_to_lower_res = []
        self.mr_ref_to_higher_res = []
        
        # constraints stored in lists
        self.curvature_constraints = {}
        self.user_constraints = {}
        self.next_curvature_constraint_id = 0
        self.next_user_constraint_id = 0
        self.constraints_dirty = False

        # build the high-res crossfield
        high_res_crossfield = CrossField()
        high_res_crossfield.load_from_bmesh(self.mesh)
        self.fields.append(high_res_crossfield)

        # construct multi-res hierarchy
        multires_hierarchy_timer_start = time.thread_time()
        layer = 1
        early_stop = False
        max_faces_per_merged_blob = 2 # soft barrier
        while (not early_stop) and (layer < num_layers):
            layer_merge_max = math.cos(math.radians(((num_layers-1 - layer) * 2 + (layer - 1) * max_merge_angle)) / (num_layers - 2))
            current_field : CrossField = copy.deepcopy(self.fields[-1])
            current_to_prev = {}
            prev_to_current = {}
            for cross_id in current_field.graph.keys():
                current_to_prev[cross_id] = set([cross_id])
                prev_to_current[cross_id] = cross_id
            # merge until max merge angle is reached
            first_merge_attempt = True
            while True:
                merge_candidates = []
                for cross_id, nb_ids in current_field.graph.items():
                    for nb_id in nb_ids:
                        if nb_id < cross_id:
                            continue # visit each edge once
                        if len(current_field.mesh_faces[cross_id]) > max_faces_per_merged_blob: #and len(current_field.mesh_faces[nb_id]) > max_faces_per_merged_blob:
                            continue # dont merge
                        normal_dot = current_field.normals[cross_id].dot(current_field.normals[nb_id])
                        if normal_dot < layer_merge_max:
                            continue # dont merge 
                        merge_score = normal_dot * max(current_field.areas[cross_id] / current_field.areas[nb_id], current_field.areas[nb_id] / current_field.areas[cross_id])
                        merge_candidates.append(((cross_id, nb_id), merge_score))
                # stop if no merge is available
                if len(merge_candidates) == 0:
                    if first_merge_attempt:
                        early_stop = True
                    break
                first_merge_attempt = False
                # sort the merge candidates in descending order
                merge_candidates_sorted = sorted(merge_candidates, key=lambda x: x[1], reverse=True)
                # do the merge
                merged_field, ref_to_lower_res, ref_to_higher_res = merge_crossfield(current_field, merge_candidates_sorted)
                # update current field
                current_field = merged_field # dont know if deepcopy is needed
                # update references to prev  
                __current_to_prev = {}
                for large_cross_id, refined_pair in ref_to_higher_res.items():
                    __current_to_prev[large_cross_id] = current_to_prev[refined_pair[0]].union(current_to_prev[refined_pair[1]])
                current_to_prev = copy.copy(__current_to_prev)
                for small_cross_id in prev_to_current.keys():
                    prev_to_current[small_cross_id] = ref_to_lower_res[prev_to_current[small_cross_id]]
            
            if early_stop:
                continue # hierarchy done
            # add the current field to the hierarchy
            self.fields.append(current_field)
            self.mr_ref_to_higher_res.append(current_to_prev)
            self.mr_ref_to_lower_res.append(prev_to_current)
            print("Hierarchy level added with", len(self.fields[-1].graph.keys()), "crosses.")
            layer += 1
            max_faces_per_merged_blob *= 2
        multires_hierarchy_time = time.thread_time() - multires_hierarchy_timer_start
        print("Multires Crossfield Hierarchy computed in ", multires_hierarchy_time, "seconds.")
        print("Number of layers: ", len(self.fields))
        print("Number of field reference dicts:", len(self.mr_ref_to_lower_res))

        self.compute_curvature_constraints()
        self.optimize()
        self.singularities = []
        self.calculate_singularities()

    def optimize(self, convergence_eps=1e-2, max_iterations=100):
        # Propagate constraints if necessary
        if self.constraints_dirty:
            self.propagate_constraints()

        # optimize bottom up
        for i in reversed(range(len(self.fields))):
            self.fields[i].optimize(convergence_eps, max_iterations)
            if i == 0:
                continue
            # copy directions to the next layer 
            higher_id = i - 1
            for j in self.fields[higher_id].graph.keys():
                self.fields[higher_id].crossdirs[j] = self.fields[i].crossdirs[self.mr_ref_to_lower_res[higher_id][j]]
                curr_normal : Vector = self.fields[higher_id].normals[j]
                self.fields[higher_id].crossdirs[j] -= curr_normal * curr_normal.dot(self.fields[higher_id].crossdirs[j])
                self.fields[higher_id].crossdirs[j].normalize()

    def calculate_singularities(self):
        """ Calculate singular vertices """
        self.singularities = []
        for vertex in self.mesh.verts:
            if vertex.is_boundary:
                continue # assume no singularity on boundary
            unordered_face_ring = vertex.link_faces
            face_ring = [unordered_face_ring[0]]
            # select one neighbour
            second_face_candidates = [f for e in face_ring[0].edges for f in e.link_faces if (f in unordered_face_ring) and (f != face_ring[0])]  
            face_ring.append(second_face_candidates[0])
            while face_ring[0] != face_ring[-1]:
                curr_face = face_ring[-1]
                next_face_c = [f for e in curr_face.edges for f in e.link_faces if (f != curr_face) and (f != face_ring[-2]) and (f in unordered_face_ring)]
                if len(next_face_c) == 0:
                    # handle special case
                    face_ring.append(face_ring[0])
                    continue
                next_face = next_face_c[0]
                face_ring.append(next_face)    

            defekt = vertex_angle_defekt(vertex)
            # sum up angles between crosses while traversing the face ring
            angle = 0
            extrinsic_index = 0
            for i in range(len(face_ring) - 1):
                faceA = face_ring[i].index
                faceB = face_ring[i+1].index
                # curr_angle = cross_angle_intrinsic(self.fields[0].crossdirs[faceA],
                #                                    self.fields[0].normals[faceA],
                #                                    self.fields[0].crossdirs[faceB],
                #                                    self.fields[0].normals[faceB])
                curr_ind_extrinsic = cross_alignment_indices(self.fields[0].crossdirs[faceA],
                                                             self.fields[0].normals[faceA],
                                                             self.fields[0].crossdirs[faceB],
                                                             self.fields[0].normals[faceB])
                extrinsic_index += curr_ind_extrinsic[1] -  curr_ind_extrinsic[0]

            index = round(4 * (angle + defekt) / (2 * math.pi))

            # if vertex.index == 2:
            #     print("summed angles:", 4 * angle / (2 * math.pi))
            #     print("angle defekt:", 4 * defekt / (2 * math.pi))
            #     print("Index: ", index)
            #     print("Extrinsic index:", extrinsic_index % 4)
            #     self.singularities.append((vertex, index))

            if extrinsic_index % 4 != 0 and extrinsic_index % 4 != 2:
                # print("Extrinsic singularity detected!")
                # print("Index:", extrinsic_index % 4)
                self.singularities.append((vertex, extrinsic_index % 4))

            # if index != 0:
            #     print("summed angles:", angle / (2 * math.pi))
            #     print("angle defekt:", defekt / (2 * math.pi))
            #     print("Index: ", index)
            #     self.singularities.append((vertex, index))

    def compute_curvature_constraints(self, activation_threshold = 2.5, saturation_threshold = 10, max_weight = 1):
        self.clear_curvature_constraints()
        for face in self.mesh.faces:
            _principal_dir, significance = principal_curvature_dir_with_score(face)
            principal_dir = Vector(_principal_dir)
            if significance < activation_threshold:
                continue
            if significance > saturation_threshold:
                constraint_weight = max_weight
            else:
                constraint_weight = (significance - activation_threshold) * max_weight / (saturation_threshold - activation_threshold)
            self.add_curvature_constraint(face, principal_dir, constraint_weight)

    def propagate_constraints(self):
        for i in range(1, len(self.fields)):
            prev_layer_id = i-1
            self.fields[i].clear_constraints()
            for cross_index in self.fields[i].graph.keys():
                weight_sum = 0
                num_collected_constraints = 0
                constraint_dir = Vector()
                cross_normal = self.fields[i].normals[cross_index]
                # collect all constraints of merged crosses
                for higher_level_cross_id in self.mr_ref_to_higher_res[prev_layer_id][cross_index]:
                    if higher_level_cross_id not in self.fields[prev_layer_id].constraints:
                        continue
                    for higher_level_constraint in self.fields[prev_layer_id].constraints[higher_level_cross_id].values():
                        curr_dir = higher_level_constraint[0]
                        curr_weight = higher_level_constraint[1]
                        # project onto tangent plane
                        proj_dir = curr_dir - cross_normal * cross_normal.dot(curr_dir)
                        # if first constraint, just add
                        if num_collected_constraints == 0:
                            constraint_dir = proj_dir
                        else:
                            # make directions line up
                            aligned_constraint_dir, aligned_curr_dir = align_crosses(constraint_dir, cross_normal, proj_dir, cross_normal)
                            # new direction is a weighted sum
                            constraint_dir = aligned_constraint_dir * weight_sum + aligned_curr_dir * curr_weight
                        weight_sum += curr_weight
                        num_collected_constraints += 1
                if num_collected_constraints == 0:
                    continue
                # project and normalize directional constraint
                constraint_dir -= cross_normal * cross_normal.dot(constraint_dir)
                constraint_dir.normalize()
                constraint_weight = weight_sum / num_collected_constraints
                self.fields[i].add_constraint(cross_index, constraint_dir, constraint_weight)
        self.constraints_dirty = False

    def add_curvature_constraint(self, face : bmesh.types.BMFace, direction : Vector, weight):
        if face.index in self.fields[0].graph:
            added_id = self.fields[0].add_constraint(face.index, direction, weight)
            self.curvature_constraints[self.next_curvature_constraint_id] = (face.index, added_id)
            self.next_curvature_constraint_id += 1
            self.constraints_dirty = True
            return self.next_curvature_constraint_id - 1
        else:
            print("Warning! Tried to add a curvature constraint to a non existing face")
            return -1

    def del_curvature_constraint(self, constraint_id):
        if constraint_id < 0 or constraint_id not in self.curvature_constraints:
            return # invalid constraints 
        else:
            # delete constraint from top layer 
            self.fields[0].del_constraint(*self.curvature_constraints[constraint_id])
            # remove constraint 
            self.curvature_constraints.pop(constraint_id)
            self.constraints_dirty = True

    def clear_curvature_constraints(self):
        # remove all from the top hierarchy layer
        for curvature_constraint in self.curvature_constraints.values():
            self.fields[0].del_constraint(*curvature_constraint)
        self.curvature_constraints = {}
        self.next_curvature_constraint_id = 0
        self.constraints_dirty = True

    def add_user_constraint(self, face : bmesh.types.BMFace, direction : Vector, weight):
        if face.index in self.fields[0].graph:
            added_id = self.fields[0].add_constraint(face.index, direction, weight)
            self.user_constraints[self.next_user_constraint_id] = (face.index, added_id)
            self.next_user_constraint_id += 1
            self.constraints_dirty = True
            return self.next_user_constraint_id - 1
        else:
            print("Warning! Tried to add a user constraint to a non existing face")
            return -1

    def del_user_constraint(self, constraint_id):
        if constraint_id < 0 or constraint_id not in self.user_constraints:
            return # invalid constraints 
        else:
            # delete constraint from top layer 
            self.fields[0].del_constraint(*self.user_constraints[constraint_id])
            # remove constraint 
            self.user_constraints.pop(constraint_id)
            self.constraints_dirty = True

    def clear_user_constraints(self):
        # remove all from the top hierarchy layer
        for curvature_constraint in self.user_constraints.values():
            self.fields[0].del_constraint(*curvature_constraint)
        self.user_constraints = {}
        self.next_user_constraint_id = 0
        self.constraints_dirty = True

    def cross_points_for_rendering(self, level=0):
        if level < 0 or level >= len(self.fields):
            return self.fields[0].batch_ready_lines_for_crosses(self.world_matrix)
        return self.fields[level].batch_ready_lines_for_crosses(self.world_matrix)
    
    def graph_points_for_rendering(self, level=0):
        if level < 0 or level >= len(self.fields):
            return self.fields[0].batch_ready_lines_for_graph(self.world_matrix)
        return self.fields[level].batch_ready_lines_for_graph(self.world_matrix)

    def merged_faces_for_rendering(self, level=0):
        if level < 0 or level >= len(self.fields):
            return self.fields[0].batch_ready_tris_for_coloring(self.mesh, self.world_matrix)
        return self.fields[level].batch_ready_tris_for_coloring(self.mesh, self.world_matrix)

    def constraint_crosses_for_rendering(self, level=0):
        if level < 0 or level >= len(self.fields):
            return self.fields[0].batch_ready_lines_for_constraints(self.world_matrix)
        return self.fields[level].batch_ready_lines_for_constraints(self.world_matrix)

    def singularity_points_for_rendering(self):
        sing_points = {}
        for sing in self.singularities:
            coord = tuple(self.world_matrix @ sing[0].co)
            index = sing[1]
            sing_points.setdefault(index, []).append(coord)
        return sing_points

class CrossField():
    """ Internal Cross Field data structure """
    def __init__(self):
        self.clear()

    def clear(self):
        self.graph = {}
        self.normals = {}
        self.crossdirs = {}
        self.centers = {}
        self.areas = {}
        self.mesh_faces = {}
        self.constraints = {} # can contain user imposed constraints to the field as well as constraints derived from curvature
        self.next_constraint_id = 0 # store constraints by index for easy removal

    def optimize(self, convergence_eps=1e-3, max_iterations=100):
        start_time = time.thread_time()
        max_change = convergence_eps
        iterations = 0
        while max_change >= convergence_eps and iterations < max_iterations: 
            max_change = 0
            # iterate over the field and optimize locally
            for cross_id, nb_cross_ids in self.graph.items():
                n_i = self.normals[cross_id]
                weight_sum = 0
                sum = self.crossdirs[cross_id] # maybe just zero vector (as in the paper) controls how much the cross wants to change its orientation
                # iterate over all neighbouring crosses
                for nb_cross_id in nb_cross_ids:
                    n_j = self.normals[nb_cross_id]
                    cross_j = self.crossdirs[nb_cross_id]
                    w_j = 1.0 # uniform weights for now
                    # make orientations compatible
                    aligned_i, aligned_j = align_crosses(sum, n_i, cross_j, n_j)
                    # add direction of cross j to the sum
                    sum = weight_sum * aligned_i + w_j * aligned_j
                    # project onto tangent plane and normalize
                    sum -= n_i * n_i.dot(sum)
                    sum.normalize()
                    weight_sum += w_j

                # iterate over constraints
                if cross_id in self.constraints:
                    for constraint_dir, constraint_w in self.constraints[cross_id].values():
                        # make orientations compatible
                        aligned_i, aligned_constraint = align_crosses(sum, n_i, constraint_dir, n_i)
                        # add constraint direction to sum
                        sum = weight_sum * aligned_i + constraint_w * aligned_constraint
                        # project and normalize
                        sum -= n_i * n_i.dot(sum)
                        sum.normalize()
                        weight_sum += constraint_w       

                # measure cross direction change to check for convergence
                aligned_old, aligned_new = align_crosses(self.crossdirs[cross_id], n_i, sum, n_i)
                max_change = max(max_change, (aligned_old - aligned_new).length)
                # update the direction of cross i
                self.crossdirs[cross_id] = sum
            iterations += 1
        opt_time = time.thread_time() - start_time
        print("Cross Field optimization ended after", iterations, "iterations.")
        print("The optimization took", opt_time, "seconds")

    def load_from_bmesh(self, mesh : bmesh.types.BMesh):
        self.clear()
        for face in mesh.faces:
            link_face_ids = [f.index for e in face.edges for f in e.link_faces if f is not face]
            self.graph[face.index] = link_face_ids
            self.normals[face.index] = face.normal
            self.crossdirs[face.index] = Vector.orthogonal(face.normal).normalized()
            self.areas[face.index] = face.calc_area()
            self.mesh_faces[face.index] = set([face.index])
            center = Vector()
            for vert in face.verts:
                center += vert.co
            center /= len(face.verts)
            self.centers[face.index] = center

    def add_constraint(self, cross_index, dir, weight):
        self.constraints.setdefault(cross_index, dict())[self.next_constraint_id] = (dir, weight)
        self.next_constraint_id += 1
        return self.next_constraint_id - 1
    
    def del_constraint(self, cross_index, constr_id):
        if cross_index in self.constraints and constr_id in self.constraints[cross_index]:
            self.constraints[cross_index].pop(constr_id)
        else:
            print("Warning! Tried to delete non-existing constraint")

    def clear_constraints(self):
        self.constraints = {}
        self.next_constraint_id = 0

    def batch_ready_lines_for_crosses(self, final_transform_matrix):
        line_coords = []
        for cross_id in self.graph.keys():
            center = self.centers[cross_id]
            cross_size = math.sqrt(self.areas[cross_id]) / 3

            n : Vector = self.normals[cross_id]
            o : Vector = cross_size * self.crossdirs[cross_id]
            R = Matrix.Rotation(math.radians(90), 4, n)

            # create the cross
            curr_vec = o
            for _ in range(4):
                line_coords.append(center)
                line_coords.append(center + curr_vec)
                curr_vec = R @ curr_vec

        # transform to world space and make tuples
        line_coords = [tuple(final_transform_matrix @ coord) for coord in line_coords]
        return line_coords

    def batch_ready_lines_for_constraints(self, final_transform_matrix):
        line_coords = []
        for cross_id, constraint_dict in self.constraints.items():
            # take constraint average
            dir = Vector()
            summed_weights = 0
            num_added = 0
            n : Vector = self.normals[cross_id]
            for constraint in constraint_dict.values():
                curr_dir = constraint[0]
                curr_w = constraint[1]
                if num_added == 0:
                    dir = curr_dir
                else:
                    aligned_dir, aligned_curr_dir = align_crosses(dir, n, curr_dir, n)
                    dir = summed_weights * aligned_dir + curr_w * aligned_curr_dir
                summed_weights += curr_w
                num_added += 1
                dir.normalize()
            
            center = self.centers[cross_id]
            cross_size = math.sqrt(self.areas[cross_id]) / 3

            o : Vector = cross_size * dir
            R = Matrix.Rotation(math.radians(90), 4, n)

            # create the cross
            curr_vec = o
            for _ in range(4):
                line_coords.append(center)
                line_coords.append(center + curr_vec)
                curr_vec = R @ curr_vec

        # transform to world space and make tuples
        line_coords = [tuple(final_transform_matrix @ coord) for coord in line_coords]
        return line_coords

    def batch_ready_lines_for_graph(self, final_transform_matrix):
        line_coords = []
        for cross_id, nb_ids in self.graph.items():
            for nb_id in nb_ids:
                if nb_id < cross_id:
                    continue
                line_coords.append(self.centers[cross_id])
                line_coords.append(self.centers[nb_id])
        # transform to world space and make tuples
        line_coords = [tuple(final_transform_matrix @ coord) for coord in line_coords]
        return line_coords

    def batch_ready_tris_for_coloring(self, mesh : bmesh.types.BMesh, final_transform_matrix):
        vertex_coords = []
        for v in mesh.verts:
            n = v.normal
            vertex_coords.append(tuple(final_transform_matrix @ (v.co + 0.01 * n)))

        indices_per_patch = []
        face_list = list(mesh.faces)
        for cross_id in self.graph.keys():
            patch_indices = []
            for f_id in self.mesh_faces[cross_id]:
                face_verts = [v.index for v in face_list[f_id].verts]
                i = 1
                while i + 1 < len(face_verts):
                    patch_indices += [(face_verts[0], face_verts[i], face_verts[i+1])] # add triangle
                    i += 1
            indices_per_patch.append(patch_indices)
        return vertex_coords, indices_per_patch

def align_crosses(cross_A : Vector, normal_A : Vector, cross_B : Vector, normal_B : Vector):
    best_score = -math.inf
    best_A = Vector()
    best_B = Vector()
    options_A = [cross_A, normal_A.cross(cross_A)]
    options_B = [cross_B, normal_B.cross(cross_B)]
    
    for option_A in options_A:
        for option_B in options_B:
            score = abs(option_A.dot(option_B))
            if score > best_score:
                best_score = score
                best_A = option_A
                best_B = option_B

    sign_B = 1 if best_A.dot(best_B) > 0 else -1

    return best_A, sign_B * best_B

def rotate_cross_into_another_plane(cross_source : Vector, normal_source : Vector, normal_target : Vector):
    rot_axis = normal_source.cross(normal_target)
    rot_cos_angle = normal_source.dot(normal_target)
    if rot_cos_angle < 0.9999:
        return rot_cos_angle * cross_source + rot_axis.cross(cross_source) + ((1 - rot_cos_angle) / rot_axis.dot(rot_axis)) * rot_axis.dot(cross_source) * rot_axis
    return cross_source

def cross_angle_intrinsic(cross_A : Vector, normal_A : Vector, cross_B : Vector, normal_B : Vector):
    cross_B_in_A = rotate_cross_into_another_plane(cross_B, normal_B, normal_A)
    crossB90_in_A = normal_A.cross(cross_B_in_A)

    dotAB = cross_A.dot(cross_B_in_A)
    dotAB90 = cross_A.dot(crossB90_in_A)

    if abs(dotAB) > abs(dotAB90):
        alignedB = cross_B_in_A if dotAB > 0 else -cross_B_in_A
    else:
        alignedB = crossB90_in_A if dotAB90 > 0 else -crossB90_in_A

    cos_angle = cross_A.dot(alignedB)
    if cos_angle < 0.9999:
        # compute signed angle
        angle = math.acos(cos_angle)
        sign = 1 if normal_A.dot(cross_A.cross(alignedB)) > 0 else -1
        return sign * angle
    return 0

def cross_alignment_index_intrinsic(cross_A : Vector, normal_A : Vector, cross_B : Vector, normal_B : Vector):
    cross_B_in_A = rotate_cross_into_another_plane(cross_B, normal_B, normal_A)
    dotAB = cross_A.dot(cross_B_in_A)
    dotA90B = (normal_A.cross(cross_A)).dot(cross_B_in_A)

    if abs(dotAB) > abs(dotA90B):
        return 0 if dotAB > 0 else 2
    return 1 if dotA90B > 0 else 3

def cross_alignment_indices(cross_A : Vector, normal_A : Vector, cross_B : Vector, normal_B : Vector):
    best_score = -math.inf
    best_A = 0
    best_B = 0
    options_A = [cross_A, normal_A.cross(cross_A)]
    options_B = [cross_B, normal_B.cross(cross_B)]
    
    for i, option_A in enumerate(options_A):
        for j, option_B in enumerate(options_B):
            score = abs(option_A.dot(option_B))
            if score > best_score:
                best_score = score
                best_A = i
                best_B = j

    if options_A[i].dot(options_B[j]) < 0:
        best_B += 2

    return best_A, best_B

def vertex_angle_defekt(vertex : bmesh.types.BMVert):
    incident_faces = vertex.link_faces
    incident_edges = vertex.link_edges

    defekt = 2 * math.pi
    for f in incident_faces:
        interesting_edges = [e for e in f.edges if e in incident_edges]
        assert len(interesting_edges) == 2
        # collect vertices
        v0 = interesting_edges[0].verts[0] if interesting_edges[0].verts[0] != vertex else interesting_edges[0].verts[1]
        v1 = interesting_edges[1].verts[0] if interesting_edges[1].verts[0] != vertex else interesting_edges[1].verts[1]

        # print("Face", f.index, "has angle", math.acos((v0.co - vertex.co).dot(v1.co - vertex.co)))
        defekt -= math.acos((v0.co - vertex.co).dot(v1.co - vertex.co))

    return defekt

def center_after_merge(centerA : Vector, normalA : Vector, centerB : Vector, normalB: Vector):
    """ Solve derivative of Lagrangian for 0 """
    nAcA = normalA.dot(centerA)
    nAcB = normalA.dot(centerB)
    nBcB = normalB.dot(centerB)
    nBcA = normalB.dot(centerA)
    nAnB = normalA.dot(normalB)

    denom = 1 / (1 - nAnB**2 + 1e-4)
    lambda_A = 2 * (nAcB - nAcA - nAnB * (nBcA - nBcB)) * denom
    lambda_B = 2 * (nBcA - nBcB - nAnB * (nAcB - nAcA)) * denom

    return 0.5 * (centerA + centerB) - 0.25 * (normalA * lambda_A + normalB * lambda_B)

def merge_crossfield(field : CrossField, merge_candidates):
    merged_field : CrossField = CrossField()
    cross_merged = {}
    for cross_id in field.graph.keys():
        cross_merged[cross_id] = False
    new_ids = []
    ref_to_lower_res = {}
    ref_to_higher_res = {}
    for merge_candidate in merge_candidates:
        c1 = merge_candidate[0][0]
        c2 = merge_candidate[0][1]
        # only merge if both crosses have not been merged yet
        if cross_merged[c1] or cross_merged[c2]:
            continue
        cross_merged[c1] = True
        cross_merged[c2] = True
        # merging
        new_ids.append(c1)
        ref_to_lower_res[c1] = c1
        ref_to_lower_res[c2] = c1
        ref_to_higher_res[c1] = [c1, c2]
        merged_field.areas[c1] = field.areas[c1] + field.areas[c2]
        merged_field.normals[c1] = (field.normals[c1] + field.normals[c2]).normalized()
        merged_field.crossdirs[c1] = Vector.orthogonal(merged_field.normals[c1]).normalized()
        merged_field.centers[c1] = center_after_merge(field.centers[c1], field.normals[c1], field.centers[c2], field.normals[c2])
        merged_field.mesh_faces[c1] = field.mesh_faces[c1].union(field.mesh_faces[c2])
    # if any unmerged crosses are left, copy them
    for cross_id, is_merged in cross_merged.items():
        if is_merged:
            continue
        new_ids.append(cross_id)
        ref_to_lower_res[cross_id] = cross_id
        ref_to_higher_res[cross_id] = [cross_id, cross_id]
        merged_field.areas[cross_id] = field.areas[cross_id]
        merged_field.normals[cross_id] = field.normals[cross_id]
        merged_field.crossdirs[cross_id] = field.crossdirs[cross_id]
        merged_field.centers[cross_id] = field.centers[cross_id]
        merged_field.mesh_faces[cross_id] = field.mesh_faces[cross_id]
    # create graph
    for c1 in new_ids:
        c2 = ref_to_higher_res[c1][1]
        # iterate over neighbours of both c1 and c2
        combined_neighbours = set([ref_to_lower_res[nb] for nb in field.graph[c1]] + [ref_to_lower_res[nb] for nb in field.graph[c2]])
        # remove the self-adjacency
        combined_neighbours.discard(c1)
        merged_field.graph[c1] = list(combined_neighbours)

    return merged_field, ref_to_lower_res, ref_to_higher_res
