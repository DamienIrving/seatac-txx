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
from unseen import indices
import plotting_utils


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    plotting_utils.set_plot_params(args.plotparams)
    
    ds_obs = fileio.open_dataset(args.obs_file)
    all_obs_shape, all_obs_loc, all_obs_scale = indices.fit_gev(ds_obs['tasmax'].values)
    logging.info(f'All observations GEV fit: shape={all_obs_shape}, location={all_obs_loc}, scale={all_obs_scale}')
    obs_nomax_values = ds_obs['tasmax'].values[:-1]
    nomax_obs_shape, nomax_obs_loc, nomax_obs_scale = indices.fit_gev(obs_nomax_values)
    logging.info(f'Observations with max omitted GEV fit: shape={nomax_obs_shape}, location={nomax_obs_loc}, scale={nomax_obs_scale}')

    ds_raw = fileio.open_dataset(args.raw_model_file)
    ds_raw_stacked = ds_raw.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()
    raw_shape, raw_loc, raw_scale = indices.fit_gev(ds_raw_stacked['tasmax'].values, generate_estimates=True)
    logging.info(f'Model (raw) GEV fit: shape={raw_shape}, location={raw_loc}, scale={raw_scale}')

    ds_bias = fileio.open_dataset(args.bias_corrected_model_file)
    ds_bias_stacked = ds_bias.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()
    bias_shape, bias_loc, bias_scale = indices.fit_gev(ds_bias_stacked['tasmax'].values, generate_estimates=True)
    logging.info(f'Model (bias corrected) GEV fit: shape={bias_shape}, location={bias_loc}, scale={bias_scale}')

    fig = plt.figure(figsize=[10, 24])
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    bins = np.arange(23, 49)
    gev_xvals = np.arange(22, 49, 0.1)
    
    #top panel
    ds_obs['tasmax'].plot(
        ax=ax1,
        marker='o',
        color='tab:blue',
    )
    ax1.set_ylabel('temperature (C)')
    ax1.set_xlabel('year')
    ax1.set_title('(a) TXx at SeaTac: Observed timeseries')

    #middle panel
    ds_obs['tasmax'].plot.hist(
        ax=ax2,
        bins=bins,
        density=True,
        rwidth=0.9,
        alpha=0.5,
        color='tab:blue',
    )
    for year in range(len(obs_nomax_values)):
        values = np.delete(obs_nomax_values, year)
        shape, loc, scale = indices.fit_gev(values)
        pdf = gev.pdf(gev_xvals, shape, loc, scale)
        ax2.plot(gev_xvals, pdf, color='tab:gray', linewidth=1.5)
    all_obs_pdf = gev.pdf(gev_xvals, all_obs_shape, all_obs_loc, all_obs_scale)
    ax2.plot(gev_xvals, all_obs_pdf, color='tab:blue', linewidth=4.0)
    nomax_obs_pdf = gev.pdf(gev_xvals, nomax_obs_shape, nomax_obs_loc, nomax_obs_scale)
    ax2.plot(gev_xvals, nomax_obs_pdf, color='tab:blue', linestyle='--', linewidth=2.0)
    ax2.legend()
    ax2.set_xlabel('TXx (C)')
    ax2.set_ylabel('probability')
    ax2.set_title('(b) TXx at SeaTac: Observed distribution')

    #bottom panel
    ds_obs['tasmax'].plot.hist(
        ax=ax3,
        bins=bins,
        density=True,
        rwidth=0.9,
        alpha=0.5,
        color='tab:blue',
        label='Station Observations'
    )
    ax3.plot(gev_xvals, all_obs_pdf, color='tab:blue', linewidth=4.0)
    ds_bias_stacked['tasmax'].plot.hist(
        ax=ax3,
        bins=bins,
        density=True,
        rwidth=0.9,
        alpha=0.5,
        color='tab:orange',
        label='ACCESS-D'
    )
    bias_pdf = gev.pdf(gev_xvals, bias_shape, bias_loc, bias_scale)
    ax3.plot(gev_xvals, bias_pdf, color='tab:orange', linewidth=4.0)
    raw_pdf = gev.pdf(gev_xvals, raw_shape, raw_loc, raw_scale)
    ax3.plot(gev_xvals, raw_pdf, color='tab:orange', linestyle='--', linewidth=2.0)
    ax3.legend()
    ax3.set_xlabel('TXx (C)')
    ax3.set_ylabel('probability')
    ax3.set_title('(c) TXx at SeaTac: Model and observed distribution')

    infile_logs = {
        args.bias_corrected_model_file: ds_bias.attrs['history'],
        args.obs_file: ds_obs.attrs['history']
    }
    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(
        args.outfile,
        metadata={metadata_key: new_log},
        bbox_inches='tight',
        facecolor='white',
        dpi=400,
    )


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
