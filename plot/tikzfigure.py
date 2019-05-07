from core import data
from constants import ROOT_DIR
from os.path import join
import matplotlib2tikz as tikz
import matplotlib.pyplot as plt
import numpy as np
from core import lipkin_quasi_spin


def save(title, base_dir=join(ROOT_DIR, 'figures')):
    title = join(base_dir, title)
    tikz.save(title + ".tex",
              figurewidth='14cm',
              figureheight='8cm',
              textsize=11.0,
              tex_relative_path_to_data=None,
              externalize_tables=False,
              override_externals=False,
              strict=True,
              wrap=True,
              add_axis_environment=True,
              extra_axis_parameters=['font =\\footnotesize', 'scale only axis'],
              extra_tikzpicture_parameters=None,
              dpi=None,
              show_info=False,
              include_disclaimer=True,
              standalone=False,
              float_format="{:.15g}", )


###############################################################################
# Color maps
color_map_blue = np.array(
    [(161, 218, 180), (65, 182, 196), (34, 94, 168), (37, 52, 148)]) / 255

color_map_red = np.array(
    [(254, 240, 217), (253, 204, 138), (252, 141, 89), (215, 48, 31)]) / 255

color_map_gray = np.array(
    [(204, 204, 204), (150, 150, 150), (99, 99, 99), (37, 37, 37)]) / 255

###############################################################################
# Line styles
linestyles = ['-', '--', '-.', ':']
linewidth = 1
fontsize = 10
###############################################################################
