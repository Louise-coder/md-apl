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
    my_areas = []
    for region in my_voronoi.regions:
        if -1 not in region and len(region) > 0:
            region_vertices = [
                my_voronoi.vertices[vertice_index]
                for vertice_index in region
            ]
            polygon = Polygon(region_vertices)
            my_areas.append(polygon.area)
    return my_areas


def get_voronoi(p_layer: ndarray, frame_index: int, nb_frame: int):
    """Compute Voronoi tesselations on a set of coordinates.

    Parameters
    ----------
    p_layer : ndarray
        For a layer, the array containing the coordinates of P atoms.
    """
    my_voronoi = Voronoi(p_layer)
    if frame_index in (0, nb_frame - 1):
        voronoi_plot_2d(my_voronoi)
        plt.savefig("src/fig/voronoi_diagram_" + str(frame_index))
    return my_voronoi


def get_p_from_layer(all_p_atoms: AtomGroup) -> Tuple:
    """Get the coordinates of the P atoms from each layer.

    Parameters
    ----------
    all_p_atoms : AtomGroup
        For a frame, the coordinates of all P atoms.

    Returns
    -------
    Tuple(List, List)
        A tuple containing the coordinates of p atoms from both layers.
    """
    all_z_pos = all_p_atoms.positions[:, 2]
    z_avg = sum(all_z_pos) / len(all_z_pos)
    p_up = all_p_atoms.select_atoms("prop z >=" + str(z_avg))
    p_down = all_p_atoms.select_atoms("prop z <" + str(z_avg))
    return p_up, p_down


def get_frame_apl(universe: Universe, frame_index: int) -> float:
    """Given a frame, compute the apl.

    Parameters
    ----------
    universe : Universe
        The universe of interest.
    frame_index : int
        The index of the frame of interest.

    Returns
    -------
    frame_apl : float
        The computed apl.
    """
    all_p_atoms = universe.select_atoms("name P")
    p_up, _ = get_p_from_layer(all_p_atoms)
    my_voronoi = get_voronoi(
        p_up.positions[:, 0:2],
        frame_index,
        len(universe.trajectory),
    )
    my_areas = get_areas(my_voronoi)
    frame_apl = sum(my_areas) / len(my_areas)
    return frame_apl


def get_trajectory_apl(universe: Universe):
    """Get the average area per lipid value.

    Parameters
    ==========
    my_universe : Universe
        The universe of interest.
    """
    sum_frame_apl = 0
    for my_frame in universe.trajectory:
        frame_apl = get_frame_apl(universe, my_frame.frame)
        sum_frame_apl += frame_apl
    trajectory_apl = sum_frame_apl / (len(universe.trajectory) * 1000)
    return trajectory_apl


if __name__ == "__main__":
    my_universe = Universe(GRO_FILE, XTC_FILE)
    trajectory_apl = get_trajectory_apl(my_universe)
    print(trajectory_apl)
