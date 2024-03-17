"""Module computing the APL (Area Per Lipid) using Voronoi Tesselations.

Authors : Dounia BENYAKHLAF & Louise LAM
Molecular Dynamics - 2024
"""

import matplotlib.pyplot as plt
from MDAnalysis import Universe, AtomGroup
from numpy import ndarray
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, box, Point
from typing import Tuple, List

from common import (
    SINGLE_GRO_FILE,
    SINGLE_XTC_FILE,
    MIX_GRO_FILE,
    MIX_XTC_FILE,
    EXP_DMPC_VALUE,
    EXP_DMPG_VALUE,
)

CHOICE = -1


def is_inside_polygon(my_points: List, my_polygon: Polygon) -> bool:
    """Whereas one of the points is in a polygon.

    Parameters
    ----------
    my_points : List
        The list of points to verify.
    my_polygon : Polygon
        The polygon of interest.

    Returns
    -------
    bool
        If one of the points is in the polygon.
    """
    if CHOICE == 2:
        for point in my_points:
            if my_polygon.contains(Point(point)):
                return True
        return False
    else:
        return True


def get_areas(
    my_voronoi: Voronoi, box_dimensions: ndarray, dmpc_p_up: AtomGroup
) -> Tuple:
    """Compute the areas of the Voronoi Diagram's cells for DMPC and DMPG.

    Parameters
    ----------
    my_voronoi : Voronoi
        The Voronoi diagram of interest.
    box_dimensions : ndarray
        The dimensions of the simulation box.
    dmpc_p_up : AtomGroup
        The selection of DMPC P atoms.

    Returns
    -------
    dmpc_areas, dmpg_areas : Tuple(List, List)
        The computed list of areas for DMPC and DMPG lipids.
    """
    dmpc_areas, dmpg_areas = [], []
    my_box = box(0, 0, box_dimensions[0], box_dimensions[1])
    for region in my_voronoi.regions:
        if -1 not in region and len(region) > 0:
            region_vertices = [
                my_voronoi.vertices[vertice_index] for vertice_index in region
            ]
            polygon = Polygon(region_vertices)
            intersection = polygon.intersection(my_box)
            is_dmpc = is_inside_polygon(dmpc_p_up.positions[:, 0:2], intersection)
            if is_dmpc:
                dmpc_areas.append(intersection.area)
            else:
                dmpg_areas.append(intersection.area)
    return dmpc_areas, dmpg_areas


def get_voronoi(p_layer: ndarray, frame_index: int, nb_frame: int) -> Voronoi:
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
        plt.title("Voronoi diagram for the frame nb " + str(frame_index))
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


def get_frame_apl(universe: Universe, frame_index: int) -> Tuple:
    """Given a frame, compute the apl for DMPC and DMPG lipids.

    Parameters
    ----------
    universe : Universe
        The universe of interest.
    frame_index : int
        The index of the frame of interest.

    Returns
    -------
    dmpc_frame_apl, dmpg_frame_apl : Tuple(float, float)
        The computed apl in nm² for both types of lipids.

    Notes
    -----
    The conversion factor is set to 100 because 1nm² = 100Å².
    """
    all_p_atoms = universe.select_atoms("name P")
    all_p_up, _ = get_p_from_layer(all_p_atoms)
    dmpc_p_up = all_p_atoms.select_atoms("resname DMPC")
    dmpc_frame_apl, dmpg_frame_apl = -1, -1
    my_voronoi = get_voronoi(
        all_p_up.positions[:, 0:2], frame_index, len(universe.trajectory)
    )
    dmpc_areas, dmpg_areas = get_areas(my_voronoi, universe.dimensions, dmpc_p_up)
    conversion_factor = 100
    dmpc_frame_apl = sum(dmpc_areas) / (len(dmpc_areas) * conversion_factor)
    if CHOICE == 2:
        dmpg_frame_apl = sum(dmpg_areas) / (len(dmpg_areas) * conversion_factor)
    return dmpc_frame_apl, dmpg_frame_apl


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
    if type == "DMPC":
        exp_value = EXP_DMPC_VALUE
    else:
        exp_value = EXP_DMPG_VALUE
    plt.plot(x, y)
    plt.plot(x, [trajectory_apl] * len(all_frame_apl), color="red")
    plt.plot(x, [exp_value] * len(all_frame_apl), color="orange")
    plt.xlabel("Frames")
    plt.ylabel("APL (nm²)")
    plt.title(f"Variation of {type} APL values across trajectory frames")
    plt.savefig("src/fig/" + type + "_apl_distribution")
    print("\033[90m* The APL distribution has been saved in fig/\033[0m")
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
    dmpc_list_apl, dmpg_list_apl = [], []
    dmpc_apl, dmpg_apl = -1, -1
    for my_frame in universe.trajectory:
        if my_frame.frame % 100 == 0:
            print(f".......... FRAME {my_frame.frame} ..........")
        dmpc_frame_apl, dmpg_frame_apl = get_frame_apl(universe, my_frame.frame)
        dmpc_list_apl.append(dmpc_frame_apl)
        if CHOICE == 2:
            dmpg_list_apl.append(dmpg_frame_apl)
    get_apl_distribution(dmpc_list_apl, "DMPC")
    dmpc_apl = sum(dmpc_list_apl) / len(dmpc_list_apl)
    if CHOICE == 2:
        get_apl_distribution(dmpg_list_apl, "DMPG")
        dmpg_apl = sum(dmpg_list_apl) / len(dmpg_list_apl)
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
    print(f"\nNumber of atoms:\t{nb_atoms}")
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
[1] Single (MPC)\n[2] Mix (MPC and MPG)\nChoice: "
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
    CHOICE = choice
    display_info(my_universe)
    dmpc_apl, dmpg_apl = get_trajectory_apl(my_universe)
    if dmpc_apl != -1:
        print(
            f"\nComputed APL DMPC for this trajectory : \033[94m{dmpc_apl:.3f} nm²\033[0m\n"
        )
    if dmpg_apl != -1:
        print(
            f"Computed APL DMPG for this trajectory : \033[94m{dmpg_apl:.3f} nm²\033[0m\n"
        )
