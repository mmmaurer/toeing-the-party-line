# Toeing the Party Line: Election Manifestos as a Key to Understand Political Discourse on Twitter

This repository contains the code and data for the paper "Toeing the Party Line: Election Manifestos as a Key to Understand Political Discourse on Twitter".

To run any code from this repository, install the packages from `requirements.txt`.

Note that to run any of our code, we assume certain naming conventions (e.g. datasets to be named `btw{year}_tweets.csv`) and columns to be present and named accordingly. In particular, the column with the Tweet text should be called `text`. For any other meta-data necessary for replication, we provide it in the respective experiment data in this repository. 

Since we are using Polars for efficient data handling and processing, you may want to adapt the `DTYPES` dictionary in [`src/data_processing.py`](src/data_processing.py) to fit your data.
## Model availability

The models are available through the Hugging Face Hub under 
- [`mmmaurer/sbert-hashtag-german-politicians-2017`](https://huggingface.co/mmmaurer/sbert-hashtag-german-politicians-2017)
- [`mmmaurer/sbert-hashtag-german-politicians-2021`](https://huggingface.co/mmmaurer/sbert-hashtag-german-politicians-2017)

## Data availability
We provide Tweet IDs and additional meta-information necessary to replicate our experiments in [`data/`](data/).
To retrieve the individual collections, see [https://doi.org/10.4232/1.12319](https://doi.org/10.4232/1.12319) for 2013, [https://doi.org/10.4232/1.12992](https://doi.org/10.4232/1.12992) for 2017, and [https://doi.org/10.4232/1.14233](https://doi.org/10.4232/1.14233) for 2021.

# Hashtag Fine-Tuning Sentence Transformers
To finetune models as described in the paper, create training and validation datasets such that the given label for a positive example sentence pair (i.e. two sentences that have at least one common hashtag) is $1$ and of a negative example sentence pair (i.e. two sentences that do not have a co-occurring hashtag) has the label $0$.

We provide a script for gathering such datasets:

```
python3 gather_dataset.py \
        --data_path [path to your dataset dir] \
        --year_lower_bound [first year to consider for sampling Tweets] \
        --year_upper_bound [last year to consider for sampling Tweets] \
        --n_partes [Minimum number of parties a hashtag should occur for] \
        --n_tweets [Minimum number of Tweets a hashtag should occur in] \
        --year [Year of the dataset] \
        --additional_filename [Optional additional marker to append to the
                               dataset name]
```
The data (Tweet IDs and label per sentence pair) for replicating our models is available in [`data/finetuning/`](data/finetuning/).

To train a model with the resulting datasets, run

```
python3 finetune_sbert.py \
        --data_path [path to your datasets dir] \
        --year [dataset target year] \
        --output_path [path to output dir for the trained model] \
        --batch_size [batch size] \
        --epochs [epochs] \
        --warmup_steps [warmup steps] \
        --pool [ids of the GPUs in pool, separated with "-", e.g. 1-2-3] \
        --cuda [if you use a single GPU, indicate the id, e.g. 0]
```

# Replicating our experiments

Per experiment, the data (Tweet IDs and necessary meta-data) is available in [`data/`](data/) under a subdirectory with the respective experiment number (e.g. [`data/experiment1/`](data/experiment1/))

To replicate our experiments, first retrieve embeddings for your given model using 
```
python3 embeddings.py --data_dir [path to your dataset dir] \
                      --output_dir [output path for the pickled embeddings] \
                      --model [model name to gather embeddings for] \
                      --year [dataset target year] \
                      --cache_dir [path to your transformers cache dir]
```

For our robustness experiments, the resulting embeddings can be filtered using

```
python3 filter_embeddings --model [model name] \
                          --embeddings_path [path to the pickled embeddings] \
                          --data_path [path to the dataset dir] \
                          --year [dataset target year] \
                          --mode [filtering mode, see below] \
                          --month_start [start month, only used in time mode] \
                          --month_end [end month, only used in time mode] \
                          --parties [list of party abbreviations] \
                          --seed [random seed, only used in random mode] \
                          --fraction [fraction of data to keep, used in random mode]
```

For experiment 2, use mode `random` with the seeds $0$, $6$, $12$, $24$, $42$ with fractions from $0.125$ to $1.0$ in $0.125$ stepsize steps. For experiment 3 (a) mode `time` with `start_month` $1$, $3$, $6$ and `end_month` $9$, and individual months (i.e. `start_month` $1$, `end_month` $1$). For experiment 4, use `mode`s ```elected```, ```incumbent```, ```reelected```, ```newly_elected```, and ```not_reelected```.

To run the evaluation in the experiments, the manifesto dataset is needed. To download the version we used in the paper, visit [https://doi.org/10.25522/manifesto.mpds.2023a](https://doi.org/10.25522/manifesto.mpds.2023a). For the latest version, visit [https://manifesto-project.wzb.eu/datasets](https://manifesto-project.wzb.eu/datasets).

Finally, to run the experiments, use

```
python3 experiments.py --embeddingspath [path to the pickled embeddings] \
                       --manifestopath [path to the manifesto csv] \
                       --datapath [path to the dataset dir] \
                       --outputpath [path to output the results to] \
                       --year [Election year to run experiment for] \
                       --mode [experiment mode] \
                       --seed [random seed for random mode] \
                       --fraction [fraction of tweets to choose randomly] \
                       --start_month [start month for time mode] \
                       --end_month [end month for time mode]
                       --parties [list of party abbreviations]
```
The `mode` and additional mode-specific settings should be according to how the embeddings were filtered.

# Citation
To cite our work, use the GitHub's citation function or copy:
```bibtex
@inproceedings{maurer-etal-2024-toeing,
    title = "Toeing the Party Line: Election Manifestos as a Key to Understand Political Discourse on {T}witter",
    author = "Maurer, Maximilian  and
      Ceron, Tanise  and
      Pad{\'o}, Sebastian  and
      Lapesa, Gabriella",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.354",
    doi = "10.18653/v1/2024.findings-emnlp.354",
    pages = "6115--6130",
}
```