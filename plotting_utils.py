"""Plotting utilities"""

import yaml
import matplotlib as mpl


image_metadata_keys = {
    "png": "History",
    "pdf": "Title",
    "eps": "Creator",
    "ps": "Creator",
}

def set_plot_params(param_file):
    """Set the matplotlib parameters."""

    if param_file:
        with open(param_file, "r") as reader:
            param_dict = yaml.load(reader, Loader=yaml.BaseLoader)
    else:
        param_dict = {}
    for param, value in param_dict.items():
        mpl.rcParams[param] = value