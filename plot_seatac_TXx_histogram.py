"""Plot SeaTac tXx histogram"""

import pdb
import sys
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import genextreme as gev

from unseen import fileio
from unseen import general_utils
from unseen import indices


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    general_utils.set_plot_params(args.plotparams)
    
    ds_obs = fileio.open_dataset(args.obs_file)
    obs_shape, obs_loc, obs_scale = indices.fit_gev(ds_obs['tasmax'].values)
    logging.info(f'Observations GEV fit: shape={obs_shape}, location={obs_loc}, scale={obs_scale}')

    ds_raw = fileio.open_dataset(args.raw_model_file)
    ds_raw_stacked = ds_raw.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()
    raw_shape, raw_loc, raw_scale = indices.fit_gev(ds_raw_stacked['tasmax'].values, use_estimates=True)
    logging.info(f'Model (raw) GEV fit: shape={raw_shape}, location={raw_loc}, scale={raw_scale}')

    ds_bias = fileio.open_file(args.bias_corrected_model_file)
    ds_bias_stacked = ds_bias.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()
    bias_shape, bias_loc, bias_scale = indices.fit_gev(ds_bias_stacked['tasmax'].values, use_estimates=True)
    logging.info(f'Model (bias corrected) GEV fit: shape={bias_shape}, location={bias_loc}, scale={bias_scale}')

    fig = plt.figure(figsize=[10, 16])
    ax1 = fig.add_subplot(211)

    bins = np.arange(23, 49)
    gev_xvals = np.arange(22, 49, 0.1)
    
    ds_bias_stacked['tasmax'].plot.hist(
        ax=ax1,
        bins=bins,
        density=True,
        rwidth=0.9,
        alpha=0.7,
        color='tab:blue',
        label='ACCESS-D'
    )
    bias_pdf = gev.pdf(gev_xvals, bias_shape, bias_loc, bias_scale)
    ax1.plot(gev_xvals, bias_pdf, color='tab:blue', linewidth=2.0)
    raw_pdf = gev.pdf(gev_xvals, raw_shape, raw_loc, raw_scale)
    ax1.plot(gev_xvals, raw_pdf, color='tab:blue', linestyle='--', linewidth=2.0)

    ds_obs['tasmax'].plot.hist(
        ax=ax1,
        bins=bins,
        density=True,
        rwidth=0.9,
        alpha=0.7,
        color='tab:orange',
        label='Station Observations'
    )
    obs_pdf = gev.pdf(gev_xvals, obs_shape, obs_loc, obs_scale)
    ax1.plot(gev_xvals, obs_pdf, color='tab:orange', linewidth=2.0)

    ax1.legend()
    ax1.set_xlabel('TXx (C)')
    ax1.set_ylabel('probability')
    ax1.set_title('Histogram of TXx: SeaTac')

    infile_logs = {
        args.bias_corrected_model_file: ds_bias.attrs['history'],
        args.obs_file: ds_obs.attrs['history']
    }
    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = fileio.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("obs_file", type=str, help="Observations data file")
    parser.add_argument("raw_model_file", type=str, help="Model file (raw)")
    parser.add_argument("bias_corrected_model_file", type=str, help="Model file (bias corrected)")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension')
    
    args = parser.parse_args()
    _main(args)
