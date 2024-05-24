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
    '%(asctime)s %(levelname)-8s [zeroish] %(message)s'
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def get_feature_cutoffs_percentile(f_mat, percentile):
    f_percentile_cutoffs = pd.Series(
        f_mat.apply(
            lambda r: 10**np.percentile(
                np.log10(r[
                    r.nonzero()
                ]),
                percentile
            ),
            axis=0,
            engine='numba',
            raw=True,
            engine_kwargs={
                'parallel': True,
            },
        ),
        index=f_mat.columns
    )
    return f_percentile_cutoffs

def generate_pdet_matrix(count_mat, total_counts_vec, feature_cutoffs_vec):
    zero_values = pd.DataFrame(
        np.where(count_mat == 0),
        index=['obs_i', 'feat_j']
    ).astype(int).T
    zero_values.index = range(len(zero_values))
    p_det = zero_values.apply(
        lambda r: np.exp(
            -1*total_counts_vec[r[0]]*feature_cutoffs_vec[r[1]]
        ),
        axis=1,
        raw=True,
        engine='numba',        
        engine_kwargs={
            'parallel': True,
        },
    )
    p_det_mat = np.ones(
        shape=count_mat.shape,
        dtype=np.float32
    )
    for idx, row in zero_values.iterrows():
        p_det_mat[row.obs_i, row.feat_j] = p_det[idx]

    return p_det_mat    

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
            'csv',
            ','
        )
    elif clipped_fn.endswith('.tsv'):
        return (
            fh,
            'tsv',
            '\t'
        )
    elif clipped_fn.endswith('.txt'):
        return(
            fh,
            'tsv',
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
    args_parser.add_argument(
        '--output-layer', '-OL',
        help='(for anndata output only) layer into which to place the probability of detection (defaut: prob_detected)',
        default="prob_detected",
        type=str,
    )
    

    args = args_parser.parse_args()
    logging.info("Attempting to identify input filetype.")
    # Open the file using autodetection
    (fh, filetype, delimiter) = autodetect_input_and_open(args.input)
    logging.info(
        f'{args.input} was identified as being of type {filetype} with delimiter {delimiter}.'
    )
    if filetype == 'anndata':
        d = ad.read_h5ad(args.input)
        if args.input_layer == "":
            if scipy.sparse.issparse(d.X):
                logging.warn(
                    "You have provided sparse data. The probability-detected matrix to be generated will be dense. This may result in excessive memory use."
                )
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
    elif filetype == 'csv' or filetype == 'tsv':
        count_mat = pd.read_csv(
            fh,
            delimiter=delimiter,
            index_col=0,
        )

    else:
        logging.error(
            f"{args.input} is not of a format I can recognize or open (tsv, txt, csv or anndata)."
        )
        sys.exit(404)
    logging.info(
        f"Completed loading data. There are {count_mat.shape[0]} observations and {count_mat.shape[1]} features."
    )

    # Generate a fractional abundance matrix after getting the per-obs total reads
    total_counts = count_mat.sum(axis=1)
    f_mat = (count_mat.T / total_counts).T
    logging.info("Generated relative abundance matrix. Starting feature cutoff detection.")
    # Great now identify per-feature cutoffs
    feature_cutoffs_percentile = get_feature_cutoffs_percentile(
        f_mat, 
        percentile=args.percentile
    )
    logging.info(f"Generated per-feature cutoffs for {len(feature_cutoffs_percentile)} features.")
    p_det_mat = generate_pdet_matrix(
        count_mat,
        total_counts.astype(np.int32).values,
        feature_cutoffs_percentile.astype(np.float32).values,
    )
    logging.info("Completed generation of the specimen-feature probability of detection.")
    ## Output type
    if args.output is sys.stdout:
        pd.DataFrame(
            p_det_mat,
            index=count_mat.index,
            columns=count_mat.columns
        ).to_csv(sys.stdout)
        sys.exit(0)
    # Implicit else
    if args.output.lower().endswith('.csv') or args.output.lower().endswith('.tsv') or args.output.lower().endswith('.txt') \
        or args.output.lower().endswith('.csv.gz') or args.output.lower().endswith('.tsv.gz') or args.output.lower().endswith('.txt.gz'):
        if delimiter is None:
            delimiter = ','
        pd.DataFrame(
            p_det_mat,
            index=count_mat.index,
            columns=count_mat.columns
        ).to_csv(
            args.output,
            sep=delimiter,
        )
        sys.exit(0)
    # Implicit else
    if args.output.lower().endswith('.h5ad') and filetype == 'anndata':
        # I realize this may be a reload. 
        d = ad.read_h5ad(args.input)
        if args.output_layer in d.layers:
            logging.warn(
                f"Layer {args.output_layer} exists in the provided ann data and will be overwritten."
            )
        d.layers[args.output_layer] = p_det_mat
        d.write_h5ad(
            args.output
        )
        sys.exit(0)
    # Implicit else
    if args.output.lower().endswith('.h5ad'):
        # Requesting anndata output without anndata input. That's fine.
        d = ad.AnnData(
            p_det_mat,
            obs=pd.DataFrame(index=count_mat.index),
            var=pd.DataFrame(index=count_mat.columns)
        )
        d.layers[args.output_layer] = p_det_mat
        d.write_h5ad(
            args.output
        )
        sys.exit(0)
    
    logging.info("Done.")

if __name__ == "__main__":
    main()


