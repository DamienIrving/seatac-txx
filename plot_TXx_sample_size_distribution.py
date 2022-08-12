"""Plot SeaTac TXx sample size distribution"""

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
from unseen import indices
import plotting_utils


def _main(args):
    """Run the command line program."""

    plotting_utils.set_plot_params(args.plotparams)
    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    
    n_repeats = 1000
    fig = plt.figure(figsize=[10, 16])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Panel b: Model data
    ds_ensemble = fileio.open_dataset(args.ensemble_file)
    ds_ensemble_stacked = ds_ensemble.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()

    population_size = ds_ensemble_stacked['tasmax'].size
    maximum = float(ds_ensemble_stacked['tasmax'].max().values)
    logging.info(f'Maximum model TXx: {maximum}C')

    df_random_model = pd.DataFrame([maximum]*n_repeats, columns=[population_size])
    for sample_size in [10, 50, 100, 500, 1000, 5000, 10000]:
        estimates = []
        for resample in range(n_repeats):
            random_indexes = np.random.choice(population_size, size=sample_size, replace=False)
            #random_indexes.sort()
            tasmax_max = float(ds_ensemble_stacked['tasmax'].isel({'sample': random_indexes}).max().values)
            estimates.append(tasmax_max)
        df_random_model[sample_size] = estimates
    df_random_model = df_random_model.reindex(sorted(df_random_model.columns), axis=1)

    df_random_model.boxplot(ax=ax2)
    ax2.set_title('(b) Maximum TXx from model ensemble')
    ax2.set_xlabel('sample size')
    ax2.set_ylabel('TXx (C)')

    # Panel a: Obs
    ds_obs = fileio.open_dataset(args.obs_file)
    obs_shape, obs_loc, obs_scale = indices.fit_gev(ds_obs['tasmax'].values)
    logging.info(f'Observations GEV fit: shape={obs_shape}, location={obs_loc}, scale={obs_scale}')
    df_random_obs = pd.DataFrame()
    for sample_size in [10, 50, 100, 500, 1000, 5000, 10000, population_size]:
        estimates = []
        for resample in range(n_repeats):
            gev_data = gev.rvs(
                obs_shape,
                loc=obs_loc,
                scale=obs_scale,
                size=sample_size
            )
            txx_max = gev_data.max()
            estimates.append(txx_max)
        df_random_obs[sample_size] = estimates
    df_random_obs.boxplot(ax=ax1)
    ax1.axhline(42.2, linestyle='--', color='0.5')
    ax1.set_title('(a) Maximum TXx from observations GEV')
    ax1.set_xlabel('sample size')
    ax1.set_ylabel('TXx (C)')   
        
    infile_logs = {args.ensemble_file : ds_ensemble.attrs['history']}
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=sys.path[0])
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ensemble_file", type=str, help="Model ensemble TXx file")
    parser.add_argument("obs_file", type=str, help="Observations TXx file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension')
    args = parser.parse_args()
    _main(args)
