# Zeroish

## Usage:
`zeroish -I <input_file> -O <output_file>`

## The problem it solves:

Compositional data can be generated from a variety of sources, including (single-cell) RNA-sequencing, 16S or whole genome shotgun microbiome data, and others. In biological systems, a log-normal distribution is typical for the relative contributions of each feature in a given specimen. A few features will comprise the bulk of the counts for a specimen, with a long and often deep tail. Some of the most functionally important features (such as transcription factors, or methanogens in RNA or microbiomes respectively) are only found at very low levels relative to other features. Further, it typical that the total observations per specimen will vary--often by almost an order of magnitude, even within the same batch of specimens processed with the same methods. 

This results in a variable limit of detection between specimens that often shadows over some of the most functionally important features. Sticking with the example of a critical transcription factor (e.g., `LGR5` in the intestine as the critical transcription factor for intestinal stem cells) in single-cell RNA sequencing, is is expected that only a handful (1-5) reads will be assigned to `LGR` out of 10,000 total reads when `LGR` is expressed. If 'on' is only a few reads out of thousands, how trusted can zero recovered reads be to determine that  `LGR5` was *not* expressed in that cell? For each observation and each feature, what is the probability if that feature being expressed at its 'typical' level in that observation (even if zero counts were assigned)?

## How zeroish solves this problem:

The probability if observing zero reads if the *actual* relative abundance was $m$ is:

$p(0|m) = e^{-n_{total} * m}$

where $p(0|m)$ is  of $n_{total}$ total reads.

This is in turn derived from the poisson distribution:
$p(k | n_{total}) = e^{-n_{total} * m} * \frac{{n_{total} * m}^k}{k!}$ and $k = 0$.

Using these concepts, `zeroish`:
- Accepts a raw specimen-feature-count matrix.
- Generates the relative / fractional abundance matrix.
- For each feature, *when detected* determines the log-transformed distribution of relative abundances observed.
- Sets a per feature expected minimum relative observation rate as n percentile of this distribution (default: 2.5)
- Uses the total counts per specimen combined with the expected minimum relative observation rate to establish the probability there *was* actually that feature in this observation even though zero counts observed.
- Returns a probability-detected matrix.

## Inputs:
Can be:
- `.csv` (comma-separated values), `.tsv` (tab separated), or `.txt` (whitespace delimited) specimen-feature-count matricies.
These *must* be in strict 'wide' format, with one header column of features, one row per observation with the first column observation IDs.
These can be gzipped, as indicated with a `.gz` file suffix.

OR 

- Anndata in `.h5ad` format with either `.X` or a layer containing the raw untransformed count data.
- Anndata can be `sparse` matricies but will always return a dense `probability_detected` matrix.

## Outputs:
- `.csv` `.tsv` or `.txt`. If `.gz` at the end, these will be gzipped as well.
- `.h5ad` for AnnData output. If provided an `.h5ad` AnnData as input, the probability matrix will be added as a layer to this existing object and then saved as a new anndata.

## Who to blame
Jonathan Golob, MD PhD. 
j-dev@golob.org


