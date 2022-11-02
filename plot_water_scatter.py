"""Plot hottest SeaTac day in observations and models"""

import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import logging

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from unseen import fileio
import plotting_utils


def process_point(ds_txx, ds_summer_water, rmse_df, init_date_index, lead_time_index, ensemble_index):
    """Process single data point"""
    
    init_year = str(ds_txx['init_date'].values[init_date_index].year)
    init_month = str(ds_txx['init_date'].values[init_date_index].month).zfill(2)
    init_day = str(ds_txx['init_date'].values[init_date_index].day).zfill(2)

    forecast = f'/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-{init_year}{init_month}{init_day}/atmos_isobaric_daily.zarr.zip'

    init_selection = rmse_df['forecast'] == forecast
    ensemble_selection = rmse_df['ensemble'] == ensemble_index + 1
    lead_selection = rmse_df['time'].dt.year == int(init_year) + lead_time_index + 1

    rmse_df_subset = rmse_df[init_selection & ensemble_selection & lead_selection]
    rmse_selection = rmse_df_subset.loc[rmse_df_subset['tasmax'].idxmax()]
    #rmse_selection = rmse_df_subset.iloc[(rmse_df_subset['tasmax'] - txx).abs().argsort(), :].iloc[0]
    
    txx = float(ds_txx['tasmax'].isel({
        'ensemble': ensemble_index,
        'init_date': init_date_index,
        'lead_time': lead_time_index
    }).values)
    water = float(ds_summer_water['water'].isel({
        'ensemble': ensemble_index,
        'init_date': init_date_index,
        'lead_time': lead_time_index
    }).values)
    
    #assert np.isclose(rmse_selection['tasmax'], txx)
    if not np.isclose(rmse_selection['tasmax'], txx):
        rmse_tasmax = rmse_selection['tasmax']
        logging.info(f"forecast: {forecast}, init date index: {init_date_index}, lead time index: {lead_time_index}, ensemble index: {ensemble_index}; mismatch between RMSE tasmax {rmse_tasmax} and TXx {txx}")
    rmse = rmse_selection['rmse']
    
    return txx, water, rmse


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    plotting_utils.set_plot_params(args.plotparams)
    
    ds_summer_water = xr.open_dataset(args.summer_water_file, engine='zarr')
    # The first lead time for May starts comes from an incomplete year (and needs to be removed)
    # and the last lead time from Nov starts.
    ds_summer_water_may = ds_summer_water.sel({'init_date': ds_summer_water['init_date'].dt.month == 5})
    ds_summer_water_may = ds_summer_water_may.isel({'lead_time': slice(1, 10)})
    ds_summer_water_nov = ds_summer_water.sel({'init_date': ds_summer_water['init_date'].dt.month == 11})
    ds_summer_water_nov = ds_summer_water_nov.isel({'lead_time': slice(0, 9)})

    ds_txx = xr.open_dataset(args.txx_file, engine='zarr')
    ds_txx_may = ds_txx.sel({'init_date': ds_summer_water['init_date'].dt.month == 5})
    ds_txx_nov = ds_txx.sel({'init_date': ds_summer_water['init_date'].dt.month == 11})
    
    rmse_df = pd.read_csv(args.rmse_file)
    rmse_df['time'] = pd.to_datetime(rmse_df['time'])
    
    rmse_list = []
    txx_list = []
    water_list = []
    n_init_dates, n_lead_times, n_ensembles = ds_txx_may['tasmax'].shape
    for init_date_index in range(n_init_dates):
        print('init date:', init_date_index)
        for lead_time_index in range(n_lead_times):
            print('lead time:', lead_time_index)
            for ensemble_index in range(n_ensembles):
                txx_may, water_may, rmse_may = process_point(
                    ds_txx_may,
                    ds_summer_water_may,
                    rmse_df,
                    init_date_index,
                    lead_time_index,
                    ensemble_index,
                )
                txx_list.append(txx_may)
                water_list.append(water_may)
                rmse_list.append(rmse_may)
            
                txx_nov, water_nov, rmse_nov = process_point(
                    ds_txx_nov,
                    ds_summer_water_nov,
                    rmse_df,
                    init_date_index,
                    lead_time_index,
                    ensemble_index,
                )
                txx_list.append(txx_nov)
                water_list.append(water_nov)
                rmse_list.append(rmse_nov)
            
    rmse_ds = pd.Series(rmse_list)
    txx_ds = pd.Series(txx_list)
    water_ds = pd.Series(water_list)

    df = pd.concat([txx_ds, water_ds, rmse_ds], join='inner', axis=1)
    df.columns = ['txx', 'water', 'rmse']
    
    fig = plt.figure(figsize=[8, 6])
    ax = plt.axes()

    x = np.array(rmse_list)
    y = np.array(txx_list)

    plt.scatter(
        x,
        y,
        c=water_list,
        edgecolors='black',
        linewidths=0.1,
        cmap='cividis_r'
    )
    cbar = plt.colorbar()
    cbar.set_label('summer mean surface water (kg/m2)')
    sns.kdeplot(data=df, x='rmse', y='txx', color='0.3', linewidths=0.7)

    plt.xlabel('RMSE (m)')
    plt.ylabel('TXx (C)')
    
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
    parser.add_argument("summer_water_file", type=str, help="summer average surface water file")
    parser.add_argument("txx_file", type=str, help="TXx file")
    parser.add_argument("rmse_file", type=str, help="RMSE file")
    parser.add_argument("outfile", type=str, help="output file")

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
