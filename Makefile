.PHONY: all tasmax-obs tasmax-forecast tasmax-bias-correction calc-txx-obs calc-txx-forecast calc-txx-forecast-bias-corrected plot-histogram plot-hot-day plot-sample-size-dist plot-likelihoods plot-return-periods plot-annual-max plot-distribution clean help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python
PYTHON_PATCHWORK=/g/data/xv83/dbi599/miniconda3/envs/patchwork/bin/python
PLOT_PARAMS=plotparams_publication.yml

## tasmax-obs : preparation of observed tasmax data
tasmax-obs : ${OBS_TASMAX_FILE}
${OBS_TASMAX_FILE} : ${OBS_DATA} ${OBS_METADATA}
	fileio $< $@ --metadata_file $(word 2,$^) --variables tasmax --units ${UNITS} --no_leap_day --input_freq D 

## tasmax-forecast : preparation of CAFE tasmax data
tasmax-forecast : ${FCST_TASMAX_FILE}
${FCST_TASMAX_FILE} : ${FCST_METADATA}
	fileio ${FCST_DATA} $@ --forecast --metadata_file $< --variables tasmax --units ${UNITS} --no_leap_day --input_freq D --spatial_coords ${LAT} ${LON} --output_chunks lead_time=50 --dask_config ${DASK_CONFIG}

## water-forecast : preparation of CAFE water data
water-forecast : ${FCST_WATER_FILE}
${FCST_WATER_FILE} : ${WATER_METADATA}
	fileio ${WATER_DATA} $@ --forecast --metadata_file $< --variables water --input_freq M --spatial_coords ${LAT} ${LON} --time_agg mean --season JJA --time_freq Q-NOV --reset_times

## tasmax-bias-correction : bias correct tasmax data using observations
tasmax-bias-correction : ${FCST_TASMAX_BIAS_CORRECTED_FILE}
${FCST_TASMAX_BIAS_CORRECTED_FILE} : ${FCST_TASMAX_FILE} ${OBS_TASMAX_FILE}
	bias_correction $< $(word 2,$^) tasmax ${BIAS_METHOD} $@ --base_period ${BASE_PERIOD}

## calc-txx-obs : calculate annual daily maximum temperature from observational data
calc-txx-obs : ${OBS_TXX_FILE}
${OBS_TXX_FILE} : ${OBS_TASMAX_FILE}
	fileio $< $@ --variables tasmax --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --input_freq D 

## calc-txx-forecast : calculate annual daily maximum temperature from forecast data
calc-txx-forecast : ${FCST_TXX_FILE}
${FCST_TXX_FILE} : ${FCST_TASMAX_FILE}
	fileio $< $@ --variables tasmax --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods --input_freq D 

## calc-txx-forecast-bias-corrected : calculate annual daily maximum temperature from bias corrected forecast data
calc-txx-forecast-bias-corrected : ${FCST_TXX_BIAS_CORRECTED_FILE}
${FCST_TXX_BIAS_CORRECTED_FILE} : ${FCST_TASMAX_BIAS_CORRECTED_FILE}
	fileio $< $@ --variables tasmax --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods --input_freq D 

## similarity-test : similarity test between observations and bias corrected forecast
#similarity-test : ${SIMILARITY_FILE}
#${SIMILARITY_FILE} : ${FCST_BIAS_FILE} ${OBS_PROCESSED_FILE}
#	similarity $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD}

## independence-test : independence test for different lead times
#independence-test : ${INDEPENDENCE_PLOT}
#${INDEPENDENCE_PLOT} : ${FCST_BIAS_FILE}
#	independence $< ${VAR} $@ ${INDEPENDENCE_OPTIONS}

## plot-histogram : plot TXx histogram
plot-histogram : ${TXX_HISTOGRAM_PLOT}
${TXX_HISTOGRAM_PLOT} : ${OBS_TXX_FILE} ${FCST_TXX_FILE} ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_seatac_TXx_histogram.py $< $(word 2,$^) $(word 3,$^) $@  --plotparams ${PLOT_PARAMS}

## plot-hot-day : plot hottest day in model and obs
# Get model files, ensembles and dates from find_hot_days.ipynb
plot-hot-day : ${HOT_DAY_PLOT}
${HOT_DAY_PLOT} : ${REANALYSIS_HGT_FILE} ${REANALYSIS_TAS_FILE} ${FCST_METADATA}
	${PYTHON} plot_hottest_day.py $< $(word 2,$^) 3 2 $(word 3,$^) $@ --plotparams ${PLOT_PARAMS} --point ${LON} ${LAT} --model_files ${FCST_HOT_DAY_FILES} --ensemble_numbers ${FCST_HOT_DAY_ENSEMBLES} --dates ${FCST_HOT_DAY_DATES}

## plot-sample-size-dist : plot TXx sample size distribution
plot-sample-size-dist : ${TXX_SAMPLE_PLOT}
${TXX_SAMPLE_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE} ${OBS_TXX_FILE}
	${PYTHON} plot_TXx_sample_size_distribution.py $< $@ --obs_file $(word 2,$^) --plotparams ${PLOT_PARAMS}

## plot-likelihoods : plot TXx likelihoods
plot-likelihoods : ${TXX_LIKELIHOOD_PLOT}
${TXX_LIKELIHOOD_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_TXx_likelihoods.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-return-periods : plot TXx return periods
plot-return-periods : ${TXX_RETURN_PERIODS_PLOT}
${TXX_RETURN_PERIODS_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_TXx_return_periods.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-by-year : plot maximum TXx and distribution by year
plot-by-year : ${TXX_BY_YEAR_PLOT}
${TXX_BY_YEAR_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_model_TXx_by_year.py $< $@ --plotparams ${PLOT_PARAMS}

## calc-z500-rmse : perform z500 RMSE analysis
calc-z500-rmse : ${Z500_RMSE_FILE}
${Z500_RMSE_FILE} : ${FCST_METADATA}
	${PYTHON} z500_pattern-analysis.py ${FCST_DATA} $< $@ --txxmax_file ${FCST_HOT_DAY_FILE} --metric rmse

## calc-z500-corr : perform z500 pattern correlation analysis
calc-z500-corr : ${Z500_CORR_FILE}
${Z500_CORR_FILE} : ${FCST_METADATA}
	${PYTHON} z500_pattern-analysis.py ${FCST_DATA} $< $@ --txxmax_file ${FCST_HOT_DAY_FILE} --metric corr --anomaly

## plot-z500 : plot z500 pattern analysis
plot-z500 : ${Z500_PLOT}
${Z500_PLOT} : ${Z500_RMSE_FILE} ${Z500_CORR_FILE}
	${PYTHON_PATCHWORK} plot_z500_pattern-analysis.py $< $(word 2,$^) $@ --plotparams ${PLOT_PARAMS}

## plot-water-z500 : plot z500 vs water analysis
plot-water-z500 : ${WATER_Z500_PLOT}
${WATER_Z500_PLOT} : ${FCST_WATER_FILE}  ${FCST_TXX_FILE} ${Z500_RMSE_FILE}
	${PYTHON} plot_water_scatter.py $< $(word 2,$^) $(word 3,$^) $@

## clean : remove all generated files
clean :
	rm ${TXX_HISTOGRAM_PLOT} ${REANALYSIS_HOT_DAY_PLOT} ${MODEL_HOT_DAY_PLOT} ${TXX_SAMPLE_PLOT} ${TXX_LIKELIHOOD_PLOT} ${TXX_RETURN_PERIODS_PLOT} ${TXX_ANNUAL_MAX_PLOT} ${TXX_ANNUAL_DIST_PLOT}

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

