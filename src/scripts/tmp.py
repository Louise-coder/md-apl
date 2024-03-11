import MDAnalysis as mda
from common import GRO_FILE, XTC_FILE

if __name__ == "__main__":
    my_universe = mda.Universe(GRO_FILE)
    my_trajectory = mda.coordinates.XTC.XTCReader(XTC_FILE)