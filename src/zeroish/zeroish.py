#!/usr/bin/env python3

import argparse
import numpy as np
import logging
import sys
import gzip

# Set up logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
logFormatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s [phylotypes] %(message)s'
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

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
            'txt',
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
        default=sys.stdout
    )
    
    args = args_parser.parse_args()

    (fh, filetype, delimiter) = autodetect_input_and_open(args.input)

    print(filetype)


if __name__ == "__main__":
    main()


