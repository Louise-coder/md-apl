"""Module computing the APL (Area Per Lipid) using Voronoi Tesselations.

Authors : Dounia BENYAKHLAF & Louise LAM
Molecular Dynamics - 2024
"""

import matplotlib.pyplot as plt
from MDAnalysis import Universe, AtomGroup
from numpy import ndarray
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, box
from typing import Tuple, List

from common import SINGLE_GRO_FILE, SINGLE_XTC_FILE, MIX_GRO_FILE, MIX_XTC_FILE


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
                my_voronoi.vertices[vertice_index] for vertice_index in region
            ]
            region_out = is_region_out(region_vertices, box_dimensions)
            if not region_out:
                polygon = Polygon(region_vertices)
                my_areas.append(polygon.area)
    return my_areas


def get_areas_2(my_voronoi: Voronoi, box_dimensions: ndarray) -> List:
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
    my_box = box(0, 0, box_dimensions[0], box_dimensions[1])
    for region in my_voronoi.regions:
        if -1 not in region and len(region) > 0:
            region_vertices = [
                my_voronoi.vertices[vertice_index] for vertice_index in region
            ]
            polygon = Polygon(region_vertices)
            intersection = polygon.intersection(my_box)
            my_areas.append(intersection.area)
    return my_areas


def get_voronoi(
    p_layer: ndarray, frame_index: int, nb_frame: int, type: str
) -> Voronoi:
    """Compute Voronoi tesselations on a set of coordinates.

    Parameters
    ----------
    p_layer : ndarray
        For a layer, the array containing the coordinates of P atoms.
    type : str
        The type of lipid.

    Returns
    -------
    my_voronoi : Voronoi
        The resulting Voronoi diagram.
    """
    my_voronoi = Voronoi(p_layer)
    if frame_index in (0, nb_frame - 1):
        voronoi_plot_2d(my_voronoi)
        plt.title(f"{type} Voronoi diagram for the frame nb " + str(frame_index))
        plt.savefig("src/fig/" + type + "_voronoi_diagram_" + str(frame_index))
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


def get_frame_apl(
    universe: Universe, p_atoms: AtomGroup, frame_index: int, type: str
) -> float:
    """Given a frame, compute the apl.

    Parameters
    ----------
    universe : Universe
        The universe of interest.
    p_atoms : AtomGroup
        The selected P atoms.
    frame_index : int
        The index of the frame of interest.
    type : str
        The type of lipid.

    Returns
    -------
    frame_apl : float
        The computed apl in nm2.

    Notes
    -----
    The conversion factor is set to 100 because 1nm² = 100Å².
    """
    p_up, _ = get_p_from_layer(p_atoms)

    my_voronoi = get_voronoi(
        p_up.positions[:, 0:2], frame_index, len(universe.trajectory), type
    )
    my_areas = get_areas_2(my_voronoi, universe.dimensions)
    conversion_factor = 100
    frame_apl = sum(my_areas) / (len(my_areas) * conversion_factor)
    return frame_apl


def get_apl_distribution(all_frame_apl: List, type: str):
    """Get apl distribution depending on the frame of the trajectory.

    Parameters
    ----------
    all_frame_apl : List
        The list of the apl values for each frame.
    type : str
        The type of lipid.
    """
    x = [i for i in range(len(all_frame_apl))]
    y = all_frame_apl
    trajectory_apl = sum(all_frame_apl) / len(all_frame_apl)
    plt.plot(x, y)
    plt.plot(x, [trajectory_apl] * len(all_frame_apl), color="red")
    plt.xlabel("Frames")
    plt.ylabel("APL (nm²)")
    plt.title(f"Variation of {type} APL values across trajectory frames")
    plt.savefig("src/fig/" + type + "_apl_distribution")
    print("\033[90m* The APL distribution has been saved in fig/\033[0m\n")
    plt.close()


def get_trajectory_apl(universe: Universe) -> Tuple:
    """Get the average area per lipid value.

    Parameters
    ----------
    my_universe : Universe
        The universe of interest.

    Returns
    -------
    dmpc_apl, dmpg_apl : Tuple
        The DMPC and DMPG apl in a tuple.
    """
    dmpc_frame_apl = []
    dmpg_frame_apl = []
    dmpc_apl, dmpg_apl = -1, -1
    for my_frame in universe.trajectory:
        dmpc_p_atoms = universe.select_atoms("resname DMPC and name P")
        dmpg_p_atoms = universe.select_atoms("resname DMPG and name P")
        if dmpc_p_atoms:
            frame_apl = get_frame_apl(universe, dmpc_p_atoms, my_frame.frame, "DMPC")
            dmpc_frame_apl.append(frame_apl)
        if dmpg_p_atoms:
            frame_apl = get_frame_apl(universe, dmpg_p_atoms, my_frame.frame, "DMPG")
            dmpg_frame_apl.append(frame_apl)
    if dmpc_frame_apl:
        get_apl_distribution(dmpc_frame_apl, "DMPC")
        dmpc_apl = sum(dmpc_frame_apl) / len(dmpc_frame_apl)
    if dmpg_frame_apl:
        get_apl_distribution(dmpg_frame_apl, "DMPG")
        dmpg_apl = sum(dmpg_frame_apl) / len(dmpg_frame_apl)
    return dmpc_apl, dmpg_apl


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
    print(f"Number of atoms:\t{nb_atoms}")
    print(f"Number of frames:\t{nb_frames}")
    print(f"Box dimensions:\t\t{box_dimensions}\n")
    print(f"\033[92mLoading...\033[0m\n")


def get_choice():
    """Get the choice of type membrane from the user.

    Returns
    -------
    int_choice : int
        The user choice.
    """
    is_ok = False
    int_choice = -1
    while not is_ok:
        choice = input(
            "Which type of membrane do you want to compute the apl from?\n\
[1] Single (MPC)\n[2] Mix (MPC and MPG)\n"
        )
        try:
            int_choice = int(choice)
            if int_choice in (1, 2):
                is_ok = True
        except ValueError:
            print("Invalid choice! Please enter a valid integer.")
    return int_choice


if __name__ == "__main__":
    print("\n\033[95m========== APL Calculator ==========\033[0m")
    choice = get_choice()

    if choice == 1:
        my_universe = Universe(SINGLE_GRO_FILE, SINGLE_XTC_FILE)
    else:
        my_universe = Universe(MIX_GRO_FILE, MIX_XTC_FILE)
    display_info(my_universe)
    dmpc_apl, dmpg_apl = get_trajectory_apl(my_universe)
    if dmpc_apl != -1:
        print(
            f"Computed APL DMPC for this trajectory : \033[94m{dmpc_apl:.3f} nm²\033[0m\n"
        )
    if dmpg_apl != -1:
        print(
            f"Computed APL DMPG for this trajectory : \033[94m{dmpg_apl:.3f} nm²\033[0m\n"
        )
