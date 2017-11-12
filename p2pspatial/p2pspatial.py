from __future__ import absolute_import, division, print_function
import numpy as np
import pulse2percept as p2p
import skimage.measure as skim

from .due import due, Doi

__all__ = ["SpatialSimulation", "get_region_props"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='p2pspatial')


class SpatialSimulation(p2p.Simulation):

    def set_ganglion_cell_layer(self):
        pass

    def pulse2percept(self, electrode):
        pass


def get_region_props(img):
    regions = skim.regionprops(skim.label(img))
    return regions[0] if len(regions) == 1 else regions
