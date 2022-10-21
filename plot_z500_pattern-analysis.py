"""Plot z500 RMSE and pattern correlation"""

import pdb
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import patchworklib as pw
import cmdline_provenance as cmdprov

import plotting_utils


def _main(args):
    """Run the command line program."""

    pw.overwrite_axisgrid()
    plotting_utils.set_plot_params(args.plotparams)
    
    df_rmse = pd.read_csv(args.rmse_file)
    df_corr = pd.read_csv(args.corr_file)
        
    g1 = sns.jointplot(
        data=df_rmse,
        x='RMSE (m)',
        y='Tmax (C)',
        kind='reg',
        xlim=(-5, 250),
        marginal_kws={'bins': 20},
        scatter_kws={'alpha': 0.2},
        fit_reg=False,
    )
    g1.plot_joint(sns.kdeplot, color="tab:cyan", zorder=10, levels=11)
    g1 = pw.load_seaborngrid(g1)
    
    g2 = sns.jointplot(
        data=df_corr,
        x='pattern correlation',
        y='Tmax (C)',
        kind='reg',
        marginal_kws={'bins': 20},
        scatter_kws={'alpha': 0.2},
        fit_reg=False,
    )
    g2.plot_joint(sns.kdeplot, color="tab:cyan", zorder=10, levels=11)
    g2 = pw.load_seaborngrid(g2)
    
    metadata_key = plotting_utils.image_metadata_keys[args.outfile.split('.')[-1]]
    (g1|g2).savefig(
        args.outfile,
        metadata={metadata_key: cmdprov.new_log()},
        bbox_inches='tight',
        facecolor='white',
        dpi=300,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rmse_file", type=str, help="Input RMSE csv file")
    parser.add_argument("corr_file", type=str, help="Input pattern correlation csv file")
    parser.add_argument("outfile", type=str, help="Output figure file")
    parser.add_argument(
        '--plotparams',
        type=str,
        default=None,
        help='matplotlib parameters (YAML file)'
    )
    args = parser.parse_args()
    _main(args)
