import argparse
import os
import pickle

import mantel
import numpy as np
import polars as pl

from src.data_processing import (load_data,
                                 get_hashtags,
                                 get_month,
                                 get_common_hashtags,
                                 get_representations_per_hashtag)
from src.distance import hashtag_distance_matrix
from src.evaluation import manifesto_distance_matrix

def preprocess_data(data_path,
                    mode,
                    election_year,
                    random_seed,
                    fraction,
                    month_start,
                    month_end):
    df = load_data(data_path, election_year)
    df = get_hashtags(df)

    if mode == "random":
        model =  f"{model}-{random_seed}"
        np.random.seed(random_seed)
        df = df.with_columns(
            pl.lit(np.random.choice([True, False], len(df),
                                    p=[fraction, 1-fraction])). \
                                        alias("filter_column")
        )
        df = df.filter(pl.col("filter_column")==True)
    
    elif mode == "time":
        df = get_month(df)
        df = df.filter((pl.col('month') <= month_end) & \
                       (pl.col("month") >= month_start))

    elif mode == "not_reelected":
        df = df.filter(
            (pl.col("elected") == 0) & \
                (pl.col("incumbent") == 1)
        )
    elif mode == "reelected":
        df = df.filter(
            (pl.col("elected") == 1) & \
                (pl.col("incumbent") == 1)
        )
    elif mode == "newly_elected":
        df = df.filter(
            (pl.col("elected") == 1) & \
                (pl.col("incumbent") == 0)
        )
    
    return df

def get_mode_marker(mode,
                    random_seed,
                    month_start,
                    month_end,
                    fraction):
    if mode == "full":
        marker = "full"
    elif mode == "random":
        marker = f"random-seed{random_seed}-{fraction}"
    elif mode == "time":
        marker = f"time-{month_start, month_end}"
    else:
        marker = mode
    
    return marker

def run_experiment(embeddings_path,
                   output_path,
                   manifesto_path,
                   data_path,
                   election_year,
                   mode,
                   random_seed,
                   month_start,
                   month_end,
                   fraction,
                   parties
                   ):
    # naming and path preparation
    naming = str(os.path.basename(embeddings_path)).replace("./pkl").split("-")
    model = "-".join(naming[:len(naming)-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # result writing preparation
    resultfile = os.path.join(output_path, f"results.csv")
    # if resultfile doesn't exist, create it and write header
    if not os.path.isfile(resultfile):
        with open(resultfile, "w+") as f:
            f.write("model,year,mode,eval,"
                    "r_pearson,p_pearson,z_pearson"
                    "r_spearman,p_spearman,z_spearman\n")
    marker = get_mode_marker(mode,
                             random_seed,
                             month_start,
                             month_end,
                             fraction)

    # load data
    df = preprocess_data(data_path,
                         mode,
                         election_year,
                         random_seed,
                         fraction,
                         month_start,
                         month_end)
    # load embeddings
    embeddings = pickle.load(open(embeddings_path, "rb"))
    # gather hashtags that occur across all parties and at least 50x
    common_hashtags = get_common_hashtags(df)

    # filter embeddings
    embeddings_per_hashtag = get_representations_per_hashtag(embeddings,
                                                             df,
                                                             parties)
    filtered_embeddings_per_hashtag = {}
    for tag, embs in embeddings_per_hashtag.items():
        assert len(embs)==len(parties), "Number of parties and len of" + \
            "embeddings per party should be equal."
        if tag in common_hashtags:
            filtered_embeddings_per_hashtag[tag] = embeddings_per_hashtag[tag]

    # load ground truth
    manifesto_matrix = manifesto_distance_matrix(manifesto_path,
                                                 election_year,
                                                 parties)
    
    # calculate Twitter distance matrix
    twitter_matrix = hashtag_distance_matrix(
        embeddings_per_hashtag=filtered_embeddings_per_hashtag,
        parties=parties
    )
    
    # calculate results
    r_pearson, p_pearson, z_pearson = mantel.test(twitter_matrix,
                                                  manifesto_matrix,
                                                  method='pearson')
    r_spearman, p_spearman, z_spearman = mantel.test(twitter_matrix,
                                                     manifesto_matrix,
                                                     method='spearman')

    # write results
    with open(resultfile, "a+") as f:
        f.write(
            f"{model},{election_year},{marker},full_manifesto_matrix,"
            f"{r_pearson},{p_pearson},{z_pearson},"
            f"{r_spearman},{p_spearman},{z_spearman}\n"
        )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddingspath", help="Path to the pickled"
                        " model embeddings")
    parser.add_argument("--manifestopath", help="Path to manifesto csv")
    parser.add_argument("--datapath", help="Path to dataset dectory")
    parser.add_argument("--outputpath", help="Output directory path")
    parser.add_argument("--year", help="Election year")
    parser.add_argument("--mode", help="Mode of the experiment")
    parser.add_argument("--seed", help="Seed for random choice of"
                        " tweets", default=42, type=int)
    parser.add_argument("--fraction", help="Fraction of tweets to"
                        " choose randomly", default=0.5, type=float)
    parser.add_argument("--start_month", help="Start month for time"
                        " experiment", default=6, type=int)
    parser.add_argument("--end_month", help="End month for time"
                        " experiment", default=9, type=int)
    parser.add_argument("--parties", help="parties to consider",
                        nargs="+", 
                        default=["AfD", "CDU/CSU", "FDP",
                                 "GRÜNE", "DIE LINKE", "SPD"])
    
    
    args = parser.parse_args()

    run_experiment(embeddings_path=args.embeddingspath,
                   output_path=args.outputpath,
                   data_path=args.datapath,
                   manifesto_path=args.manifestopath,
                   election_year=args.year,
                   mode=args.mode,
                   seed=args.seed,
                   fraction=args.fraction,
                   month_start=args.start_month,
                   month_end=args.end_month)

