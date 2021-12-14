"""Plot SeaTac TXx return periods"""

import pdb
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import genextreme as gev

from unseen import fileio
from unseen import general_utils
from unseen import indices


def return_period(data, score):
    """Calculate the return period for a given score"""
    
    n_exceedance_events = (data > score).sum()
    exceedance_probability = n_exceedance_events / len(data)
    
    return 1. / exceedance_probability


def plot(fname, model_subsample, gev_shape, gev_loc, gev_scale):
    """Plot one sample"""

    fig, ax = plt.subplots(figsize=[10, 8])
    bins = np.arange(23, 49)
    gev_xvals = np.arange(22, 49, 0.1)
    model_subsample.plot.hist(bins=bins,
                              density=True,
                              rwidth=0.9,
                              alpha=0.7,
                              color='tab:blue')
    gev_pdf = gev.pdf(gev_xvals, gev_shape, gev_loc, gev_scale)
    plt.plot(gev_xvals, gev_pdf, color='tab:blue', linewidth=2.0)
    plt.savefig(fname, bbox_inches='tight', facecolor='white')


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    general_utils.set_plot_params(args.plotparams)
    
    ds_ensemble = fileio.open_dataset(args.ensemble_file)
    ds_ensemble_stacked = ds_ensemble.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()

    population_size = ds_ensemble_stacked['tasmax'].size
    threshold = 42.2
    n_repeats = 1000

    full_model_return_period = return_period(ds_ensemble_stacked['tasmax'].values, threshold)
    logging.info(f'TXx={threshold}C return period in full model ensemble: {full_model_return_period}')

    full_gev_shape, full_gev_loc, full_gev_scale = indices.fit_gev(ds_ensemble_stacked['tasmax'].values, generate_estimates=True)
    full_gev_data = gev.rvs(full_gev_shape, loc=full_gev_loc, scale=full_gev_scale, size=args.gev_samples)
    full_gev_return_period = return_period(full_gev_data, threshold)
    logging.info(f'TXx={threshold}C return period from GEV fit to full model ensemble: {full_gev_return_period}')

    df_model_return_period = pd.DataFrame([full_model_return_period]*n_repeats, columns=[population_size])
    df_gev_return_period = pd.DataFrame([full_gev_return_period]*n_repeats, columns=[population_size])

    for sample_size in [10, 50, 100, 500, 1000, 5000, 10000]:
        print(sample_size)
        model_estimates = []
        gev_estimates = []
        for resample in range(n_repeats):
            gev_shape = 100
            while gev_shape > 1.0:
                random_indexes = np.random.choice(population_size, size=sample_size, replace=False)
                #random_indexes.sort()
                model_subsample = ds_ensemble_stacked['tasmax'].isel({'sample': random_indexes})
                gev_shape, gev_loc, gev_scale = indices.fit_gev(model_subsample.values, user_estimates=[full_gev_loc, full_gev_scale])
            model_return_period = return_period(model_subsample.values, threshold)
            model_estimates.append(model_return_period)
            gev_data = gev.rvs(gev_shape, loc=gev_loc, scale=gev_scale, size=args.gev_samples)  
            gev_return_period = return_period(gev_data, threshold)
            gev_estimates.append(gev_return_period)
            if args.plot:
                if resample < 10:
                    fname = f'plot_sample-size-{sample_size}_repeat-{resample}.png'
                    print(fname, gev_shape, gev_loc, gev_scale)
                    plot(fname, model_subsample, gev_shape, gev_loc, gev_scale)

        df_model_return_period[sample_size] = model_estimates
        df_gev_return_period[sample_size] = gev_estimates

    df_model_return_period = df_model_return_period.reindex(sorted(df_model_return_period.columns), axis=1)
    df_gev_return_period = df_gev_return_period.reindex(sorted(df_gev_return_period.columns), axis=1)

    fig, (ax1, ax2) = plt.subplots(2, figsize=[10, 12])
    df_model_return_period = df_model_return_period.replace(np.inf, np.nan)
    model_inf_count = df_model_return_period.isna().sum().to_string()
    logging.info(f'Infinite return periods (out of {n_repeats} repeats) in model samples:\n{model_inf_count}')
    df_model_return_period[10] = np.nan
    df_model_return_period[50] = np.nan
    df_model_return_period[100] = np.nan
    df_model_return_period.boxplot(ax=ax1)
    ax1.set_title('(a) Return periods from model samples')
    ax1.set_xlabel(' ')
    ax1.set_ylabel('return period for TXx=42.2C (years)')

    df_gev_return_period = df_gev_return_period.replace(np.inf, np.nan)
    gev_inf_count = df_gev_return_period.isna().sum().to_string()
    logging.info(f'Infinite return periods (out of {n_repeats} repeats) in GEV samples:\n{gev_inf_count}')
    df_gev_return_period[10] = np.nan
    df_gev_return_period[50] = np.nan
    df_gev_return_period[100] = np.nan
    df_gev_return_period.boxplot(ax=ax2)
    ax2.set_title('(b) Return periods from GEV fits to model samples')
    ax2.set_xlabel('sample size')
    ax2.set_ylabel('return period for TXx=42.2C (years)')
    ax2.set_ylim(-100, 2100)

    infile_logs = {args.ensemble_file : ds_ensemble.attrs['history']}
    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = fileio.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("ensemble_file", type=str, help="Model ensemble file")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension)')
    parser.add_argument('--gev_samples', type=int, default=10000,
                        help='number of times to sample the GEVs')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot some of the samples')
    
    args = parser.parse_args()
    _main(args)
