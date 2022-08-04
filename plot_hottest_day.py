"""Plot hottest SeaTac day in observations and models"""

import pdb
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


def get_max_indices(infile, config_file, point, time_bounds):
    """Get the time and ensemble index for hottest day at SeaTac"""

    ds = fileio.open_dataset(
        infile,
        variables=['tasmax'],
        metadata_file=config_file,
        spatial_coords=point,
        sel={'time': time_bounds}
    )
    argmax = ds['tasmax'].argmax(dim=['ensemble', 'time'])

    time_idx = int(argmax['time'].values)
    date = ds['time'].values[time_idx].strftime('%Y-%m-%d')
    logging.info(f'Max temperature at SeaTac, date: {date}')

    ens_idx = int(argmax['ensemble'].values)
    ensemble_member = ds['ensemble'].values[ens_idx]
    logging.info(f'Max temperature at SeaTac, ensemble member: {ensemble_member}')

    max_temp = float(ds['tasmax'].isel({'ensemble': ens_idx , 'time': time_idx}).values)
    max_temp = max_temp - 273.15
    logging.info(f'Maximum temperature at SeaTac: {max_temp}C')

    return time_idx, ens_idx


def plot_usa(ax, da_tasmax, da_h500, data_source, point=None):
    """Plot map of USA

    Args:
      da_tasmax (xarray DataArray) : maximum temperature data
      da_h500 (xarray DataArray) : 500hPa geopotential height data
      data_source (str) : data source for title
      point (list) : coordinates of point to plot (lon, lat)
    """
    
    if data_source == 'Observations':
        title_prefix = 'a'
    elif data_source == 'Model':
        title_prefix = 'b'
    else:
        raise ValueError('Unrecognised data source')

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
    ax.set_title(f'({title_prefix}) Hottest day: {data_source}')


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    plotting_utils.set_plot_params(args.plotparams)
    
    #reanalysis data
    ds_hgt = xr.open_dataset(args.obs_hgt_file, engine='cfgrib')
    da_h500 = ds_hgt['z'].mean('time')
    da_h500 = da_h500 / 9.80665
    ds_tas = xr.open_dataset(args.obs_tas_file, engine='cfgrib')
    da_tasmax = ds_tas['t2m'].max('time')
    da_tasmax = da_tasmax - 273.15

    #model data
    time_bounds = slice(f'{args.model_year}-01-01', f'{args.model_year}-12-31')
    time_idx, ens_idx = get_max_indices(args.model_file, args.model_config, args.point, time_bounds) 
    ds = fileio.open_dataset(
        args.model_file,
        variables=['h500', 'tasmax'],
        metadata_file=args.model_config,
        sel={'time': time_bounds}
    )
    ds_max = ds.isel({'ensemble': ens_idx, 'time': time_idx})
    ds_max['tasmax'] = general_utils.convert_units(ds_max['tasmax'], 'C')
    ds_max = ds_max.compute()

    fig = plt.figure(figsize=[12, 13])
    map_proj = ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=[38.5, 38.5]
    )
    ax1 = fig.add_subplot(211, projection=map_proj)
    ax2 = fig.add_subplot(212, projection=map_proj)

    plot_usa(ax1, da_tasmax, da_h500, 'Observations', point=args.point) 
    plot_usa(ax2, ds_max['tasmax'], ds_max['h500'], 'Model', point=args.point)

    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(repo_dir=repo_dir)
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]

    plt.savefig(
        args.outfile,
        metadata={metadata_key: new_log},
        bbox_inches='tight',
        facecolor='white'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("obs_hgt_file", type=str, help="reanalysis geopotential height file")
    parser.add_argument("obs_tas_file", type=str, help="reanalysis temperature file")
    parser.add_argument("model_file", type=str, help="model data file")
    parser.add_argument("model_config", type=str, help="model configuration file")
    parser.add_argument("model_year", type=str, help="model_year that the max TXx occurs in")
    parser.add_argument("outfile", type=str, help="output file")
    
    parser.add_argument('--point', type=float, nargs=2, metavar=('lon', 'lat'),
                        default=None, help='plot marker at this point')
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension')
    
    args = parser.parse_args()
    _main(args)
