#!/usr/bin/env python3

import argparse
import numpy as np
import logging
import sys
import gzip
import anndata as ad
import pandas as pd
import scipy

# Set up logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
logFormatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s [phylotypes] %(message)s'
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def per_feature_cutoffs(count_mat, percentile):
    print (count_mat)
    return

def autodetect_input_and_open(filename):
    filename =  filename.strip()
    if filename.lower().endswith('.h5ad'):
        filetype = 'anndata'
        fh = None
        delimiter = None
        return (
            fh,
            filetype,
            delimiter
        )

    # Implicit else
    if filename.lower().endswith('.gz'):
        clipped_fn = filename.lower().replace('.gz', "")
        fh = gzip.open(filename, 'rt')
    else:
        clipped_fn = filename.lower()
        fh = open(filename, 'rt')
    
    if clipped_fn.endswith('.csv'):
        return (
            fh,
            'delimited_text',
            ','
        )
    elif clipped_fn.endswith('.tsv'):
        return (
            fh,
            'delimited_text',
            '\t'
        )
    elif clipped_fn.endswith('.txt'):
        return(
            fh,
            'delimited_text',
            '\s+'
        )
    else:
        logging.warn(
            f"Could not autodetect the filetype of {filename}. Assuming it is not compressed and whitespace delimited at our peril."
        )
        return(
            fh,
            'unknown',
            '\s+'
        )
    

def main():
    args_parser = argparse.ArgumentParser(
        description="""From count-based compositional data determine if zeroes are really zeroes
        accounting for the typical frequency of features as well as total observations.
        """
    )
    args_parser.add_argument(
        '--input', '-I',
        help='The composoitional count data (will attempt to autodetect filetype)',
        required=True,
    )
    args_parser.add_argument(
        '--output', '-O',
        help='Where to store the probability matrix. (Default: stdout)',
        default=sys.stdout,
    )
    args_parser.add_argument(
        '--percentile', '-P',
        help='Percentile cutoff to be considered present. (Default 2.5)',
        default=2.5
    )
    args_parser.add_argument(
        '--input-layer', '-IL',
        help='(for anndata input only) which layer contains the raw counts',
        default="",
        type=str,
    )
    

    args = args_parser.parse_args()

    # Open the file using autodetection
    (fh, filetype, delimiter) = autodetect_input_and_open(args.input)
    if filetype == 'anndata':
        d = ad.read_h5ad(args.input)
        if args.input_layer == "":
            if scipy.sparse.issparse(d.X):
                count_mat = pd.DataFrame.sparse.from_spmatrix(
                    d.X,
                    index=d.obs_names,
                    columns=d.var_names,
                )
            else:
                count_mat = pd.DataFrame(
                    d.X,
                    index=d.obs_names,
                    columns=d.var_names,
                    dtype=np.int32
                )
        else:
            if args.input_layer not in d.layers:
                logging.error(
                    f"Layer {args.input_layer} not in {args.input}."
                )
                sys.exit(404)
            # Implicit else
            if scipy.sparse.issparse(d.layers[args.input_layer]):
                count_mat = pd.DataFrame.sparse.from_spmatrix(
                    d.layers[args.input_layer],
                    index=d.obs_names,
                    columns=d.var_names,
                )
            else:
                count_mat = pd.DataFrame(
                    d.layers[args.input_layer],
                    index=d.obs_names,
                    columns=d.var_names,
                    dtype=np.int32
                )
    elif filetype == 'delimited_text':
        count_mat = pd.read_csv(
            fh,
            delimiter=delimiter,
        )

    else:
        logging.error(
            f"{args.input} is not of a format I can recognize or open (tsv, txt, csv or anndata)."
        )
        sys.exit(404)

    # Great now identify per-feature cutoffs
    per_feature_cutoffs(
        count_mat, 
        percentile=args.percentile
    )

if __name__ == "__main__":
    main()


