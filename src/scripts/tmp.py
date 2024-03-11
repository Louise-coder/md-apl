"""Module computing the APL (Area Per Lipid) using Voronoi Tesselations.

Authors : Dounia BENYAKHLAF & Louise LAM
Molecular Dynamics - 2024
"""

from typing import Tuple
from MDAnalysis import Universe, AtomGroup
from numpy import ndarray
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

from common import GRO_FILE, XTC_FILE


def get_voronoi(p_layer: ndarray):
    """Compute Voronoi tesselations on a set of coordinates.

    Parameters
    ==========
    p_layer : ndarray
        For a layer, the array containing the coordinates of P atoms.
    """
    my_voronoi = Voronoi(p_layer)
    voronoi_plot_2d(my_voronoi)
    plt.show()


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
    for frame in universe.trajectory:
        all_p_atoms = universe.select_atoms("name P")
        p_up, _ = get_p_from_layer(all_p_atoms)
        get_voronoi(p_up.positions[:, 0:2])
        break


if __name__ == "__main__":
    my_universe = Universe(GRO_FILE, XTC_FILE)
    temporary_name(my_universe)
