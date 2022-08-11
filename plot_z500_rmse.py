"""Plot z500 pattern RMSE"""

import pdb
import sys
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import xskillscore as xs
import pandas as pd
import seaborn as sns

from unseen import fileio
from unseen import spatial_selection
from unseen import general_utils
import plotting_utils


def find_max_z500(da_tasmax, da_h500_box):
    """Find z500 corresponding to hottest day"""
    
    max_tasmax = float(da_tasmax.max(dim=['time', 'ensemble']).values)
    max_time_idx = int(da_tasmax.argmax(dim=['time', 'ensemble'])['time'].data)
    max_time_date = da_tasmax['time'].values[max_time_idx].strftime("%Y-%m-%d")
    max_ensemble_idx = int(da_tasmax.argmax(dim=['time', 'ensemble'])['ensemble'].data)
    max_tasmax_from_idx = float(da_tasmax.isel({'time': max_time_idx, 'ensemble': max_ensemble_idx}).values)
    assert max_tasmax == max_tasmax_from_idx
    log_text = f'Hottest day of {max_tasmax} on {max_time_date}, ensemble member {max_ensemble_idx + 1}'
    logging.info(log_text)
    print(log_text)
    max_z500 = da_h500_box.isel({'time': max_time_idx, 'ensemble': max_ensemble_idx})
    
    return max_z500


def is_jja(month):
    return (month >= 6) & (month <= 8)


def _main(args):
    """Run the command line program."""

    plotting_utils.set_plot_params(args.plotparams)
    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    
    lat = 47.45
    lon = 237.69
    box = [
        lat - args.distance,
        lat + args.distance,
        lon - args.distance,
        lon + args.distance
    ]
    
    ensemble_max_tasmax = 0
    files_to_search = [args.txxmax_file] if args.txxmax_file else args.infiles
    for infile in files_to_search:
        ds = fileio.open_dataset(
            infile,
            variables=['h500', 'tasmax'],
            metadata_file=args.model_config,
        )
        da_tasmax = spatial_selection.select_point_region(ds['tasmax'], [lat, lon])
        da_tasmax = general_utils.convert_units(da_tasmax, 'C')
        max_tasmax = float(da_tasmax.max(dim=['time', 'ensemble']).values)
        if max_tasmax > ensemble_max_tasmax:
            print(infile)
            da_h500_box = spatial_selection.select_box_region(ds['h500'], box)
            max_z500 = find_max_z500(da_tasmax, da_h500_box)
            ensemble_max_tasmax = max_tasmax
    
    metric_ds = pd.Series([])
    tasmax_ds = pd.Series([])
    for infile in args.infiles:
        print(infile)
        ds = fileio.open_dataset(
            infile,
            variables=['h500', 'tasmax'],
            metadata_file=args.model_config,
        )
        da_tasmax = spatial_selection.select_point_region(ds['tasmax'], [lat, lon])
        da_tasmax = general_utils.convert_units(da_tasmax, 'C')
        da_h500_box = spatial_selection.select_box_region(ds['h500'], box)
        da_tasmax_jja = da_tasmax.sel(time=is_jja(ds['time.month']))
        da_h500_box_jja = da_h500_box.sel(time=is_jja(ds['time.month']))
        if args.metric == 'rmse':
            metric_jja = xs.rmse(
                max_z500,
                da_h500_box_jja,
                dim=['lat', 'lon'],
                weights=None,
                skipna=False,
                keep_attrs=True
            )
        elif args.metric == 'corr':
            metric_jja = xs.pearson_r(
                max_z500,
                da_h500_box_jja,
                dim=['lat', 'lon'],
                weights=None,
                skipna=False,
                keep_attrs=True
            )
        metric_ds = metric_ds.append(pd.Series(metric_jja.values.flatten()), ignore_index=True)
        tasmax_ds = tasmax_ds.append(pd.Series(da_tasmax_jja.values.flatten()), ignore_index=True)

    print(len(metric_ds))
    df_list = [metric_ds, tasmax_ds]
    metric_label = 'RMSE' if args.metric == 'rmse' else 'pattern correlation'
    xlim = (-5, 250) if args.metric == 'rmse' else None
    headers = [metric_label, 'Tmax (C)']
    df = pd.concat(df_list, join='inner', axis=1)
    df.columns = headers
    g = sns.jointplot(
        data=df,
        x=metric_label,
        y='Tmax (C)',
        kind='reg',
        xlim=xlim,
    #    joint_kws={'line_kws':{'color': 'tab:cyan'}},
        marginal_kws={'bins': 20},
        scatter_kws={'alpha': 0.2},
        fit_reg=False,
    )
    g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=10, levels=8)

    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(repo_dir=repo_dir)
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(
        args.outfile,
        metadata={metadata_key: new_log},
        bbox_inches='tight',
        facecolor='white',
        dpi=300,
    )

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("infiles", type=str, nargs='*', help="Input files")
    parser.add_argument("model_config", type=str, help="model configuration file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument(
        '--plotparams',
        type=str,
        default=None,
        help='matplotlib parameters (YAML file)'
    )
    parser.add_argument(
        '--distance',
        type=float,
        default=30,
        help='distance (in degrees of lat/lon) analysis box should extend from SeaTac airport'
    )
    parser.add_argument(
        '--logfile',
        type=str,
        default=None,
        help='name of logfile (default = same as outfile but with .log extension'
    )
    parser.add_argument(
        '--txxmax_file',
        type=str,
        default=None,
        help='file containing the hottest day'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='rmse',
        choices=('rmse', 'corr'),
        help='plot the RMSE or pattern correlation'
    )
    args = parser.parse_args()
    _main(args)