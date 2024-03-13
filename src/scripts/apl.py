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


def is_region_out(region_vertices: List, box_dimensions: ndarray) -> bool:
    """Verify if the region of interest is open or not.

    Parameters
    ----------
    region_vertices : List
        The list containing the vertices of the region of interest.
    box_dimensions : ndarray
        The dimensions of the simulation box.

    Returns
    -------
    bool
        Whereas the region is open or not.
    """
    box_max_x = box_dimensions[0]
    box_max_y = box_dimensions[1]
    for x, y in region_vertices:
        if x < 0 or y < 0 or x > box_max_x or y > box_max_y:
            return True
    return False


def get_areas(my_voronoi: Voronoi, box_dimensions: ndarray) -> List:
    """Compute the areas of the Voronoi Diagram's cells.

    Parameters
    ----------
    my_voronoi : Voronoi
        The Voronoi diagram of interest.
    box_dimensions : ndarray
        The dimensions of the simulation box.

    Returns
    -------
    List
        The computed list of areas.
    """
    my_areas = []
    for region in my_voronoi.regions:
        if -1 not in region and len(region) > 0:
            region_vertices = [
                my_voronoi.vertices[vertice_index]
                for vertice_index in region
            ]
            region_out = is_region_out(region_vertices, box_dimensions)
            if not region_out:
                polygon = Polygon(region_vertices)
                my_areas.append(polygon.area)
    return my_areas


def get_voronoi(
    p_layer: ndarray, frame_index: int, nb_frame: int
) -> Voronoi:
    """Compute Voronoi tesselations on a set of coordinates.

    Parameters
    ----------
    p_layer : ndarray
        For a layer, the array containing the coordinates of P atoms.

    Returns
    -------
    my_voronoi : Voronoi
        The resulting Voronoi diagram.
    """
    my_voronoi = Voronoi(p_layer)
    if frame_index in (0, nb_frame - 1):
        voronoi_plot_2d(my_voronoi)
        plt.title("Voronoi Diagram for the frame no" + str(frame_index))
        plt.savefig("src/fig/voronoi_diagram_" + str(frame_index))
        print("\033[90m* A Voronoi diagram has been saved in fig/\033[0m")
        plt.close()
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
        The computed apl in nm2.

    Notes
    -----
    The conversion factor is set to 100 because 1nm^2 = 100A^2.
    """
    all_p_atoms = universe.select_atoms("name P")
    p_up, _ = get_p_from_layer(all_p_atoms)
    my_voronoi = get_voronoi(
        p_up.positions[:, 0:2],
        frame_index,
        len(universe.trajectory),
    )
    my_areas = get_areas(my_voronoi, universe.dimensions)
    conversion_factor = 100
    frame_apl = sum(my_areas) / (len(my_areas) * conversion_factor)
    return frame_apl


def get_apl_distribution(all_frame_apl: List):
    """Get apl distribution depending on the frame of the trajectory.

    Parameters
    ----------
    all_frame_apl : List
        The list of the apl values for each frame.
    """
    x = [i for i in range(len(all_frame_apl))]
    y = all_frame_apl
    plt.plot(x, y)
    plt.xlabel("Frames")
    plt.ylabel("APL (nm^2)")
    plt.title("Variation of APL values across trajectory frames")
    plt.savefig("src/fig/apl_distribution")
    print("\033[90m* The APL distribution has been saved in fig/\033[0m\n")
    plt.close()


def get_trajectory_apl(universe: Universe):
    """Get the average area per lipid value.

    Parameters
    ==========
    my_universe : Universe
        The universe of interest.
    """
    all_frame_apl = []
    for my_frame in universe.trajectory:
        frame_apl = get_frame_apl(universe, my_frame.frame)
        all_frame_apl.append(frame_apl)
    get_apl_distribution(all_frame_apl)
    trajectory_apl = sum(all_frame_apl) / len(all_frame_apl)
    return trajectory_apl


def display_info(universe: Universe):
    """Display the information associated to the universe.

    Parameters
    ----------
    universe : Universe
        The universe of interest.

    Notes
    -----
    You will sometimes find ANSI color codes in print statements.
    """
    nb_atoms = len(universe.atoms)
    box_dimensions = universe.dimensions[0:3]
    nb_frames = len(universe.trajectory)
    print("\n\033[95m========== APL Calculator ==========\033[0m")
    print(f"Number of atoms:\t{nb_atoms}")
    print(f"Number of frames:\t{nb_frames}")
    print(f"Box dimensions:\t\t{box_dimensions}\n")
    print(f"\033[92mLoading...\033[0m\n")


if __name__ == "__main__":
    my_universe = Universe(GRO_FILE, XTC_FILE)
    display_info(my_universe)
    trajectory_apl = get_trajectory_apl(my_universe)
    print(
        f"Computed APL for this trajectory : \033[94m{trajectory_apl:.3f} nm^2\033[0m\n"
    )
