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
import seaborn as sns

from unseen import fileio
from unseen import indices
import plotting_utils


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
    plotting_utils.set_plot_params(args.plotparams)
    
    ds_ensemble = fileio.open_dataset(args.ensemble_file)
    ds_ensemble_stacked = ds_ensemble.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()

    population_size = ds_ensemble_stacked['tasmax'].size
    threshold = 42.2

    full_model_return_period = return_period(ds_ensemble_stacked['tasmax'].values, threshold)
    logging.info(f'TXx={threshold}C return period in full model ensemble: {full_model_return_period}')

    full_gev_shape, full_gev_loc, full_gev_scale = indices.fit_gev(ds_ensemble_stacked['tasmax'].values, generate_estimates=True)
    full_gev_data = gev.rvs(full_gev_shape, loc=full_gev_loc, scale=full_gev_scale, size=args.gev_samples)
    full_gev_return_period = return_period(full_gev_data, threshold)
    logging.info(f'TXx={threshold}C return period from GEV fit to full model ensemble: {full_gev_return_period}')

    full_data = {'return_period': [full_model_return_period, full_gev_return_period],
                 'sample_size': [population_size, population_size],
                 'source': ['model samples', 'GEV fits to model samples']}
    df = pd.DataFrame(full_data)
    sample_list = [10, 50, 100, 500, 1000, 5000, 10000]
    for sample_size in sample_list:
        print(sample_size)
        for resample in range(args.n_repeats):
            gev_shape = 100
            while gev_shape > 1.0:
                random_indexes = np.random.choice(population_size, size=sample_size, replace=False)
                #random_indexes.sort()
                model_subsample = ds_ensemble_stacked['tasmax'].isel({'sample': random_indexes})
                gev_shape, gev_loc, gev_scale = indices.fit_gev(model_subsample.values, user_estimates=[full_gev_loc, full_gev_scale])
            model_return_period = return_period(model_subsample.values, threshold)
            df = df.append({'return_period': model_return_period,
                            'sample_size': sample_size,
                            'source': 'model samples'}, 
                                                        ignore_index=True)
            gev_data = gev.rvs(gev_shape, loc=gev_loc, scale=gev_scale, size=args.gev_samples)  
            gev_return_period = return_period(gev_data, threshold)
            df = df.append({'return_period': gev_return_period,
                            'sample_size': sample_size,
                            'source': 'GEV fits to model samples'},
                            ignore_index=True)
            if args.plot:
                if resample < 10:
                    fname = f'plot_sample-size-{sample_size}_repeat-{resample}.png'
                    print(fname, gev_shape, gev_loc, gev_scale)
                    plot(fname, model_subsample, gev_shape, gev_loc, gev_scale)

    df = df.replace(np.inf, np.nan)
    for sample_size in sample_list:
        model_inf_count = df['return_period'].loc[(df['source'] == 'model samples') & (df['sample_size'] == sample_size)].isna().sum()
        logging.info(f'Infinite return periods (out of {args.n_repeats} repeats) in model samples (sample size {sample_size}): {model_inf_count}')
        gev_inf_count = df['return_period'].loc[(df['source'] == 'GEV fits to model samples') & (df['sample_size'] == sample_size)].isna().sum()
        logging.info(f'Infinite return periods (out of {args.n_repeats} repeats) in GEV samples (sample size {sample_size}): {gev_inf_count}')
    df['return_period'].loc[df['sample_size'] == 10] = np.nan
    df['return_period'].loc[df['sample_size'] == 50] = np.nan
    df['return_period'].loc[df['sample_size'] == 100] = np.nan

    #sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=[10, 6])
    sns.boxplot(x='sample_size', y='return_period', hue='source', data=df)
    ax.set_title('Return periods from model ensemble')
    ax.set_xlabel('sample size')
    ax.set_ylabel('return period for TXx=42.2C (years)')
    ax.set_ylim(-100, 2100)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    ax.set_axisbelow(True)
    ax.grid(True)

    infile_logs = {args.ensemble_file: ds_ensemble.attrs['history']}
    repo_dir = sys.path[0]
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]
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
    parser.add_argument('--n_repeats', type=int, default=1000,
                        help='number of times to repeat each sample size')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot some of the samples')
    
    args = parser.parse_args()
    _main(args)
