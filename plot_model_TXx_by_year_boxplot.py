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


def plot_distribution_by_year(ax, ds_time):
    """Plot TXx distribution by year"""
    
    df = pd.DataFrame()
#    years = np.arange(2004, 2022)
    years = np.arange(1996, 2030)
    for year in years:
        year_da = ds_time['tasmax'].sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        year_array = year_da.to_masked_array().flatten().compressed()
        nsamples = year_array.shape[0]
        if nsamples < 1728:
            pad_width = 1728 - nsamples
            year_array = np.pad(year_array, (0, pad_width), constant_values=np.nan)
        logging.info(f'{nsamples} samples for the year {year}')
        df[year] = year_array
    df = pd.melt(df, var_name='year', value_name='TXx')
    sns.boxplot(data=df, x='year', y='TXx', ax=ax, palette='hot_r')
    ax.set_title('TXx distribution from model ensemble')
    ax.grid()
#    ax.set_xlabel('TXx (C)')

        
def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    plotting_utils.set_plot_params(args.plotparams)

    ds_init = xr.open_zarr(args.ensemble_file)
    ds_time = array_handling.reindex_forecast(ds_init)

    fig = plt.figure(figsize=[20, 7])
    ax1 = fig.add_subplot(111)
    
    plot_distribution_by_year(ax1, ds_time)
    
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
