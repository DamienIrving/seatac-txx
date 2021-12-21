This repository contains the details of the code, data processing steps, and software environment
associated with the following paper:

Risbey JS, Irving DB, Squire DT, Matear RJ, Monselesan DP, Pook MJ, Richardson D & Tozer CR (submitted).
On the role of weather and sampling in assessing a record-breaking heat extreme.
*Environmental Research: Climate*.

## Code, data processing steps, and software environment

The `Makefile` defines the rules/code/processing steps used to generate most of the resutls in the paper:
- Figure 2: `make plot-historgram CONFIG=workflow_config.mk`
- Figure 4: `make plot-reanalysis-hot-day CONFIG=workflow_config.mk`
- Figure 5: `make plot-model-hot-day CONFIG=workflow_config.mk`
  (see `find_hottest_model_day.ipynb` for information about the hottest day)
- Figure 7: `make plot-sample-size-dist CONFIG=workflow_config.mk`
- Figure 8: `make plot-likelihoods CONFIG=workflow_config.mk`
- Figure 9: `make plot-return-periods CONFIG=workflow_config.mk`
- Figure 10: `make plot-annual-max CONFIG=workflow_config.mk`
- Figure 11: `make plot-distribution CONFIG=workflow_config.mk`

The Python scripts called by the Makefile rely on the `unseen` package:
https://github.com/AusClimateService/unseen

The [`README`](https://github.com/AusClimateService/unseen#readme) file
for the `unseen` package describes how to install that package and
its dependencies.

## Provenance

Each output image file has the command history embedded in the image metadata.
It can be viewed by installing [exiftool](https://exiftool.org) (e.g. `$ conda install exiftool`)
and then running the following at the command line:
```bash
$ exiftool path/to/image.png
```

## Data

The observations were daily maximum temperatures at Seattle Tacoma International Airport
from the GHCNv2 station dataset,
[downloaded](http://climexp.knmi.nl/gdcntmax.cgi?id=someone@somewhere&WMO=USW00024233&STATION=SEATTLE_TACOMA_INTL_AP,_WA&extraargs=)
from the KNMI climate explorer.
