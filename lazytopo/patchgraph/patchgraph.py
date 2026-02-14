import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import namedtuple

# blender api
import bpy
from mathutils import Vector

from ..utils.constants import EPS

@dataclass
class Node:
    index : int
    position : np.ndarray
    mesh_face_index : int
    outgoing_halfedge_indices : list[int]
    deleted : bool = False

@dataclass
class Edge:
    index : int
    length : float
    subdivisions : int
    draw_dir_halfedge_index : int
    positions : np.ndarray
    mesh_face_indices : list[int]
    deleted : bool = False

@dataclass
class Halfedge:
    index : int
    edge_index : int
    opposite_halfedge_index : int
    next_halfedge_index : int
    previous_halfedge_index : int
    node_to_index : int
    node_to_is_T_junction : bool = False
    deleted : bool = False

@dataclass
class Face:
    index : int
    halfedge_index : int

SketchedEdgePoint = namedtuple("SketchedEdgePoint", ["pos_2d", "pos_3d", "face_index"])

class EndpointType(Enum):
    FLOATING = 0
    EXISTING_NODE = 1
    ON_EXISTING_EDGE = 2

@dataclass
class SketchedEdgeEndpointInfo:
    node_type : EndpointType = EndpointType.FLOATING
    node_or_edge_index : int = None
    # Where the new edge is inserted topologically:
    # EXISTING NODE: position in the halfedge ring list, so index of the next existing halfedge in cw order
    # ON_EXISTING_EDGE: whether the drawn edge splits the edges draw dir halfedge (0) or the opposite halfedge (1)
    edge_insert_position : int = None 
    edge_split_length : float = None # only used for T-junction nodes

# helper function (maybe move elsewhere later)

def _compute_sketched_edge_length(sketched_edge_points : list[SketchedEdgePoint]):
    """TODO: Compute distance on mesh surface"""
    point_array = np.array([p.pos_3d for p in sketched_edge_points])
    return np.sum(np.linalg.norm((point_array[1:,:] - point_array[0:-1,:]), axis=1))

class PatchGraph():
    def __init__(self, nodes, edges, halfedges, faces):
        self.nodes : list[Node] = nodes
        self.edges : list[Edge] = edges
        self.halfedges : list[Halfedge] = halfedges
        self.faces : list[Face] = faces

        self.max_edge_subdivisions = 50

    def get_next_indices(self):
        return len(self.nodes), len(self.edges), len(self.halfedges), len(self.faces)

    def get_opposite_halfedge(self, halfedge : Halfedge):
        return self.halfedges[halfedge.opposite_halfedge_index]

    def get_opposite_halfedge_index(self, halfedge_index : int):
        return self.halfedges[halfedge_index].opposite_halfedge_index

    def get_prev_valid_outgoing_halfedge_index(self, node_index : int, start_search_index : int):
        outgoing_halfedges = self.nodes[node_index].outgoing_halfedge_indices
        node_deg = len(outgoing_halfedges)
        search_index = (start_search_index + node_deg - 1) % node_deg
        while search_index != start_search_index:
            if not self.nodes[outgoing_halfedges[search_index]].deleted:
                break
            search_index = (search_index + node_deg - 1) % node_deg
        return outgoing_halfedges[search_index]
    
    def get_next_valid_outgoing_halfedge_index(self, node_index : int, start_search_index : int):
        outgoing_halfedges = self.nodes[node_index].outgoing_halfedge_indices
        node_deg = len(outgoing_halfedges)
        search_index = (start_search_index + 1) % node_deg
        while search_index != start_search_index:
            if not self.nodes[outgoing_halfedges[search_index]].deleted:
                break
            search_index = (search_index + 1) % node_deg
        return outgoing_halfedges[search_index]

    def add_sketched_edge(self, sketched_edge_points : list[SketchedEdgePoint],
                          target_quad_edge_length : float,
                          start_node_info : SketchedEdgeEndpointInfo = SketchedEdgeEndpointInfo(), 
                          end_node_info : SketchedEdgeEndpointInfo = SketchedEdgeEndpointInfo()):
        next_node_index, next_edge_index, next_halfedge_index, next_face_index = self.get_next_indices()
        he_0_idx = next_halfedge_index
        he_1_idx = next_halfedge_index + 1
        added_edge_length = _compute_sketched_edge_length(sketched_edge_points)
        added_edge_subdivisions = self.max_edge_subdivisions
        if target_quad_edge_length > EPS:
            added_edge_subdivisions = int(np.clip(np.round((added_edge_length / target_quad_edge_length)), 0, added_edge_subdivisions))

        # resolve new connectivity
        he_0_prev_he_idx = None
        he_1_next_he_idx = None
        he_1_node_to_idx = None
        match start_node_info.node_type:
            case EndpointType.FLOATING:
                floating_start_node = Node(index=next_node_index,
                                           position=np.array(sketched_edge_points[0].pos_3d),
                                           mesh_face_index=sketched_edge_points[0].face_index,
                                           outgoing_halfedge_indices=[next_halfedge_index])
                he_0_prev_he_idx = he_1_idx
                he_1_next_he_idx = he_0_idx
                he_1_node_to_idx = next_node_index
                self.nodes.append(floating_start_node)
                next_node_index += 1
            case EndpointType.EXISTING_NODE:
                existing_node = self.nodes[start_node_info.node_or_edge_index]                
                insert_position = start_node_info.edge_insert_position
                prev_he_idx = self.get_prev_valid_outgoing_halfedge_index(start_node_info.node_or_edge_index, insert_position)
                prev_outgoing = existing_node.outgoing_halfedge_indices

                # set values
                he_0_prev_he_idx = self.halfedges[prev_he_idx].opposite_halfedge_index
                he_1_next_he_idx = prev_outgoing[insert_position]
                he_1_node_to_idx = start_node_info.node_or_edge_index

                # we might have removed a T-junction
                self.halfedges[he_0_prev_he_idx].node_to_is_T_junction = False

                # adjust outgoing halfedges of existing node
                existing_node.outgoing_halfedge_indices = prev_outgoing[:insert_position] + [he_0_idx] + prev_outgoing[insert_position:]
            case EndpointType.ON_EXISTING_EDGE:
                existing_edge = self.edges[start_node_info.node_or_edge_index]
                existing_halfedge_0 = self.halfedges[existing_edge.draw_dir_halfedge_index]
                existing_halfedge_1 = self.get_opposite_halfedge(existing_halfedge_0)
                existing_node_0 = self.nodes[existing_halfedge_0.node_to_index]
                existing_node_1 = self.nodes[existing_halfedge_1.node_to_index]

                # split the edge at the intersection point

                # t_junction_node = Node(index=next_node_index,
                #                        position=,
                #                        mesh)

                # split the edge at the intersection point
                




            


                
        




        he_0_next_he_idx = None
        he_0_node_to_idx = None
        he_1_prev_he_idx = None




        edge_to_add = Edge(index=next_edge_index,
                           length=added_edge_length,
                           subdivisions=added_edge_subdivisions,
                           draw_dir_halfedge_index=next_halfedge_index,
                           positions=np.array([p.pos_3d for p in sketched_edge_points]),
                           mesh_face_indices=[p.face_index for p in sketched_edge_points])
        
        halfedge_1 = Halfedge(index=he_0_idx,
                              edge_index=next_edge_index,
                              opposite_halfedge_index=he_1_idx,
                              next_halfedge_index=-1,
                              previous_halfedge_index=-1,
                              node_to_index=-1,
                              node_to_is_T_junction=False)
        halfedge_2 = Halfedge(index=he_1_idx,
                              edge_index=next_edge_index,
                              opposite_halfedge_index=he_0_idx,
                              next_halfedge_index=-1,
                              previous_halfedge_index=-1,
                              node_to_index=-1,
                              node_to_is_T_junction=False)
        
        


        pass

    def remove_edge(self, edge_index):
        pass

    def remove_node(self, node_index):
        pass

    def compactify(self):
        pass
