"""Module computing the APL (Area Per Lipid) using Voronoi Tesselations.

Authors : Dounia BENYAKHLAF & Louise LAM
Molecular Dynamics - 2024
"""

import matplotlib.pyplot as plt
from MDAnalysis import Universe, AtomGroup
from numpy import ndarray
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from typing import Tuple, List

from common import GRO_FILE, XTC_FILE


def get_areas(my_voronoi: Voronoi) -> List:
    areas = []
    for region in my_voronoi.regions:
        if -1 not in region and len(region) > 0:
            region_vertices = [my_voronoi.vertices[vertice_index] for vertice_index in region]
            polygon = Polygon(region_vertices)
            areas.append(polygon.area)
    return areas


def get_voronoi(p_layer: ndarray, frame_index: int, nb_frame: int):
    """Compute Voronoi tesselations on a set of coordinates.

    Parameters
    ==========
    p_layer : ndarray
        For a layer, the array containing the coordinates of P atoms.
    """
    my_voronoi = Voronoi(p_layer)
    if(frame_index in (0, nb_frame-1)):
        voronoi_plot_2d(my_voronoi)
        plt.savefig("src/fig/voronoi_diagram_"+str(frame_index))
    return my_voronoi


def get_p_from_layer(all_p_atoms: AtomGroup) -> Tuple:
    """Get the coordinates of the P atoms from each layer.

    Parameters
    ==========
    all_p_atoms : AtomGroup
        For a frame, the coordinates of all P atoms.

    Returns
    =======
    Tuple(List, List)
        A tuple containing the coordinates of p atoms from both layers.
    """
    all_z_pos = all_p_atoms.positions[:, 2]
    z_avg = sum(all_z_pos) / len(all_z_pos)
    p_up = all_p_atoms.select_atoms("prop z >=" + str(z_avg))
    p_down = all_p_atoms.select_atoms("prop z <" + str(z_avg))
    return p_up, p_down


def temporary_name(universe: Universe):
    """Get the Voronoi Diagram for each frame of the trajectory.

    Parameters
    ==========
    my_universe : Universe
        The universe of interest.
    """
    #sum_area_per_frame = 0
    for my_frame in universe.trajectory:
        all_p_atoms = universe.select_atoms("name P")
        my_universe.atoms.wrap(compound = "atoms")
        p_up, _ = get_p_from_layer(all_p_atoms)
        print(p_up.positions)
        my_voronoi = get_voronoi(p_up.positions[:, 0:2], my_frame.frame, len(universe.trajectory))
        areas = get_areas(my_voronoi)
        avg_area = sum(areas)/len(areas)
        #sum_area_per_frame+=avg_area
        break
    #print(sum_area_per_frame/len(universe.trajectory))


if __name__ == "__main__":
    my_universe = Universe(GRO_FILE, XTC_FILE)
    temporary_name(my_universe)
