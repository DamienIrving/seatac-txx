"""Plot model maximum TXx and distribution by year"""

import pdb
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr

from unseen import fileio
from unseen import array_handling
import plotting_utils


def log_sample_counts(years, counts):
    """Log sample count for each year"""

    logging.info('Sample counts for each year:') 
    for year, count in zip(years, counts):
        logging.info(f'{year}: {count}')


def plot_max_by_year(ax, ds_time):
    """Plot maximum TXx by year"""
    
    max_by_year = ds_time['tasmax'].max(dim=('ensemble', 'init_date'), keep_attrs=True)
    max_by_year = max_by_year.resample(time='A-DEC').max('time', keep_attrs=True)

    count = ds_time['tasmax'].notnull(keep_attrs=True)
    count = count.sum(dim=('ensemble', 'init_date'), keep_attrs=True)
    count = count.resample(time='A-DEC').sum('time', keep_attrs=True)

    color = 'tab:blue'
    xvals1 = max_by_year['time'].dt.year.values
    yvals1 = max_by_year.values
    ax.set_xlabel('year')
    ax.set_ylabel('maximum TXx (C)', color=color)
    ax.plot(xvals1, yvals1, marker='o', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True)
    ax.set_ylim(40.5, 47.5)
    ax.axvspan(2004, 2021, alpha=0.5, color='0.8')

    ax_twin = ax.twinx() 

    color = '0.6'
    xvals2 = count['time'].dt.year.values
    yvals2 = count.values
    log_sample_counts(xvals2, yvals2)
    ax_twin.set_ylabel('number of samples', color=color)
    ax_twin.plot(xvals2, yvals2, color=color, linestyle='--')
    ax_twin.tick_params(axis='y', labelcolor=color)
    ax_twin.set_ylim(110, 2850)

    ax.set_title('(b) Maximum TXx from model ensemble')
    #fig.tight_layout()


def plot_distribution_by_year(ax, ds_time):
    """Plot TXx distribution by year"""
    
    years = np.arange(2004, 2022)
    color = iter(matplotlib.cm.hot_r(np.linspace(0.3, 1, len(years))))
    for year in years:
        c = next(color)
        year_da = ds_time['tasmax'].sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        year_array = year_da.to_masked_array().flatten().compressed()
        nsamples = year_array.shape[0]
        logging.info(f'{nsamples} samples for the year {year}')
        year_df = pd.DataFrame(year_array)
        sns.kdeplot(year_df[0], ax=ax, color=c, label=str(year))
    ax.grid(True)
    ax.set_xlim(26, 46)
    ax.set_title('(a) TXx distribution from model ensemble')
    ax.set_xlabel('TXx (C)')
    ax.legend(ncol=2)

        
def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    plotting_utils.set_plot_params(args.plotparams)

    ds_init = xr.open_zarr(args.ensemble_file)
    ds_time = array_handling.reindex_forecast(ds_init)

    fig = plt.figure(figsize=[12, 15])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    plot_distribution_by_year(ax1, ds_time)
    plot_max_by_year(ax2, ds_time)
    
    infile_logs = {args.ensemble_file : ds_init.attrs['history']}
    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("ensemble_file", type=str, help="Model ensemble file")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension)')
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    
    args = parser.parse_args()
    _main(args)
