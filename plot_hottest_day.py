"""Plot hottest SeaTac day in observations and models"""
import pdb
import string
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import logging

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

from unseen import fileio
from unseen import general_utils
import plotting_utils


def plot_usa(ax, da_tasmax, da_h500, title, point=None):
    """Plot map of USA

    Args:
      da_tasmax (xarray DataArray) : maximum temperature data
      da_h500 (xarray DataArray) : 500hPa geopotential height data
      title (str) : plot title
      point (list) : coordinates of point to plot (lon, lat)
    """

    h500_levels = np.arange(5000, 6300, 50)
    
    da_tasmax.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=plt.cm.hot_r,
        vmin=10,
        vmax=52,
        cbar_kwargs={'label': 'maximum temperature (C)'},
        #alpha=0.7
    )
    
    lines = da_h500.plot.contour(
        ax=ax,
        transform=ccrs.PlateCarree(),
        levels=h500_levels,
        colors=['0.1']
    )
    ax.clabel(lines, colors=['0.1'], manual=False, inline=True)
    if point:
        lon, lat = point
        ax.plot(lon, lat, 'bo', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([-140, -60, 20, 70])
    ax.gridlines(linestyle='--', draw_labels=True)
    ax.set_title(title)

    return lines


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    plotting_utils.set_plot_params(args.plotparams)
    
    ds_hgt = xr.open_dataset(args.obs_hgt_file, engine='cfgrib')
    da_h500 = ds_hgt['z'].mean('time')
    da_h500 = da_h500 / 9.80665
    ds_tas = xr.open_dataset(args.obs_tas_file, engine='cfgrib')
    da_tasmax = ds_tas['t2m'].max('time')
    da_tasmax = da_tasmax - 273.15

    nrows = int(args.nrows)
    ncols = int(args.ncols)
    if (nrows == 3) and (ncols == 2):
        figsize = [23, 20]
    else:
        raise ValueError('no figsize for that nrows/ncols combination')

    fig = plt.figure(figsize=figsize)
    map_proj = ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=[38.5, 38.5]
    )
    ax1 = fig.add_subplot(f'{nrows}{ncols}1', projection=map_proj)
    im = plot_usa(ax1, da_tasmax, da_h500, '(a) Hottest day in observations', point=args.point)

    n_model_plots = len(args.dates)
    lon, lat = args.point
    for plot_num in range(n_model_plots):
        model_file = args.model_files[plot_num]
        ensemble_number = args.ensemble_numbers[plot_num]
        date = args.dates[plot_num]
        ds = fileio.open_dataset(
            model_file,
            variables=['h500', 'tasmax'],
            metadata_file=args.model_config,
            sel={'time': date, 'ensemble': ensemble_number}
        )
        ds['tasmax'] = general_utils.convert_units(ds['tasmax'], 'C')
        print(ds['tasmax'].sel({'lat': lat, 'lon': lon}, method='nearest').values)
        ds = ds.compute()
        ax = fig.add_subplot(f'{nrows}{ncols}{plot_num+2}', projection=map_proj)

        init_date = model_file.split('/')[6].split('-')[-1]
        letter = string.ascii_lowercase[plot_num + 1]
        title = f'({letter}) Forecast {init_date}: {date}, ensemble {ensemble_number}'
        im = plot_usa(ax, ds['tasmax'], ds['h500'], title, point=args.point)

    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(repo_dir=repo_dir)
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]

    plt.savefig(
        args.outfile,
        metadata={metadata_key: new_log},
        bbox_inches='tight',
        facecolor='white',
        dpi=400,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("obs_hgt_file", type=str, help="reanalysis geopotential height file")
    parser.add_argument("obs_tas_file", type=str, help="reanalysis temperature file")
    parser.add_argument("nrows", type=str, help="number of rows for the plot")
    parser.add_argument("ncols", type=str, help="number of columns for the plot")
    parser.add_argument("model_config", type=str, help="model configuration file")
    parser.add_argument("outfile", type=str, help="output file")

    parser.add_argument(
        "--model_files",
        type=str,
        nargs='*',
        required=True,
        help="model data file for each hot day"
    )
    parser.add_argument(
        "--ensemble_numbers",
        type=int,
        nargs='*',
        required=True,
        help="ensemble numbers for each hot day"
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs='*',
        required=True,
        help="dates for each hot day"
    )
    parser.add_argument(
        '--point',
        type=float,
        nargs=2,
        metavar=('lon', 'lat'),
        default=None,
        help='plot marker at this point'
    )
    parser.add_argument(
        '--plotparams',
        type=str,
        default=None,
        help='matplotlib parameters (YAML file)'
    )
    parser.add_argument(
        '--logfile',
        type=str,
        default=None,
        help='name of logfile (default = same as outfile but with .log extension'
    )
    args = parser.parse_args()
    _main(args)
