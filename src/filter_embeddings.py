import argparse
import os
import pickle
import numpy.random as random

import polars as pl
from tqdm import tqdm

import data_processing

def get_month_filter_columns(df, month_start, month_end):
    df = data_processing.get_month(df)
    df = df.with_columns(
        ((pl.col("month") <= month_end) & \
         (pl.col("month") >= month_start)).alias("filter_column")
    )
    return df

def get_random_filter_columns(df, fraction, seed):
    random.seed(seed)
    df = df.with_columns(
        pl.lit(random.choice([True, False], len(df), p=[fraction, 1-fraction])).alias("filter_column")
    )
    return df

def get_elected_filter_columns(df):
    df = df.with_columns(
        (pl.col("elected") == 1).alias("filter_column")
    )
    return df

def get_reelected_filter_columns(df):
    df = df.with_columns(
        ((pl.col("elected") == 1) & \
         (pl.col("incumbent") == 1)
         ).alias("filter_column"),
    )
    return df

def get_newly_elected_filter_columns(df):
    df = df.with_columns(
        ((pl.col("elected") == 1) & \
         (pl.col("incumbent") == 0)
         ).alias("filter_column"),
    )
    return df

def get_not_reelected_filter_columns(df):
    df = df.with_columns(
        ((pl.col("elected") == 0) & \
         (pl.col("incumbent") == 1)
         ).alias("filter_column"),
    )
    return df

def get_content_filter_columns(df):
    df = df.with_columns(
        ((pl.col("is_retweet") == 0) & \
         (pl.col("is_quoted") == 0) & \
         (pl.col("is_reply") == 0) & \
         (pl.col("text").str.ends_with("…") == 0)
        ).alias("filter_column")
    )
    return df

def get_incumbent_filter_columns(df):
    df = df.with_columns(
        (pl.col("incumbent") == 1).alias("filter_column")
    )
    return df

def get_no_hashtags_filter_columns(df):
    df = data_processing.get_hashtags(df)
    df = df.with_columns(
        (pl.col("hashtags").list.len() == 0).alias("filter_column")
    )

def filter_embeddings(df, embeddings, parties):
    embeddings_filtered = []
    for i, party in tqdm(enumerate(parties), total=len(parties)):
        party_df = df.filter(pl.col("faction") == party)
        assert len(party_df) == len(embeddings[i]), \
            "Mismatch between embeddings and data, check order of parties"
        party_embeddings = [embeddings[i][j] for j in range(len(party_df)) if \
                            party_df["filter_column"].to_list()[j]]
        embeddings_filtered.append(party_embeddings)
    return embeddings_filtered

def filter_embeddings_random(df, embeddings, parties, p_true=0.5):
    embeddings_filtered = []
    for i, party in tqdm(enumerate(parties), total=len(parties)):
        party_df = df.filter(pl.col("faction") == party)
        filter_list = [random.choice([True, False], p=[p_true, 1-p_true]) for \
                       _ in range(len(party_df))]
        assert len(party_df) == len(embeddings[i]), \
            "Mismatch between embeddings and data, check order of parties"
        party_embeddings = [embeddings[i][j] for j in range(len(party_df)) if \
                            filter_list[j]]
        embeddings_filtered.append(party_embeddings)
    return embeddings_filtered

def save_filtered_embeddings(embeddings, model, year, month_start, month_end):
    if not os.path.exists("./embeddings/filtered"):
        os.makedirs("./embeddings/filtered")
    with open(f"./embeddings/filtered/{model}-{year}-{month_start}-"
              f"{month_end}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def save_filtered_embeddings_random(embeddings, model, year, fraction, seed):
    if not os.path.exists("./embeddings/random-choice"):
        os.makedirs("./embeddings/random-choice")
    with open(f"./embeddings/random-choice/{model}-{year}-random-{fraction}-{seed}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def save_filtered_embeddings_content(embeddings, model, year):
    if not os.path.exists("./embeddings/content"):
        os.makedirs("./embeddings/content")
    with open(f"./embeddings/content/{model}-{year}-content.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def save_filtered_embeddings_elected(embeddings, model, year):
    if not os.path.exists("./embeddings/elected"):
        os.makedirs("./embeddings/elected")
    with open(f"./embeddings/elected/{model}-"
              f"{year}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def save_filtered_embeddings_incumbent(embeddings, model, year):
    if not os.path.exists("./embeddings/incumbent"):
        os.makedirs("./embeddings/incumbent")
    with open(f"./embeddings/incumbent/{model}-{year}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def save_filtered_embeddings(embeddings, model, year, mode):
    if not os.path.exists(f"./embeddings/{mode}"):
        os.makedirs(f"./embeddings/{mode}")
    with open(f"./embeddings/{mode}/{model}-{year}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        help="Filter embeddings by time or random choice")
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="Fraction of data to keep; only used in random mode")
    parser.add_argument("--month_start", type=int, default=2,
                        help="Month to start filtering from; only used in time mode")
    parser.add_argument("--month_end", type=int, default=8,
                        help="Month to end filtering; only used in time mode")
    parser.add_argument("--parties", nargs="+",
                        default=['AfD', 'CDU/CSU', 'FDP', 'GRÜNE', 'DIE LINKE', 'SPD'])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = data_processing.load_data(args.data_path, args.year)
    embeddings = pickle.load(open(args.embeddings_path, "rb"))

    if args.mode == "time":
        df = get_month_filter_columns(df, args.month_start, args.month_end)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings(embeddings_filtered, args.model, args.year,
                                 args.month_start, args.month_end)
    elif args.mode == "random":
        df = get_random_filter_columns(df, args.fraction, args.seed)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings_random(embeddings_filtered, args.model,
                                        args.year, args.fraction, args.seed)
    elif args.mode == "content":
        df = get_content_filter_columns(df)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings_content(embeddings_filtered, args.model, args.year)
    elif args.mode == "elected":
        df = get_elected_filter_columns(df)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings_elected(embeddings_filtered, args.model, args.year)
    elif args.mode == "incumbent":
        df = get_incumbent_filter_columns(df)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings_incumbent(embeddings_filtered, args.model, args.year)
    elif args.mode == "reelected":
        df = get_reelected_filter_columns(df)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings(embeddings_filtered, args.model, args.year, "reelected")
    elif args.mode == "newly_elected":
        df = get_newly_elected_filter_columns(df)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings(embeddings_filtered, args.model, args.year, "newly-elected")
    elif args.mode == "not_reelected":
        df = get_not_reelected_filter_columns(df)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings(embeddings_filtered, args.model, args.year, "not-reelected")
    elif args.mode == "no_hashtags":
        df = get_no_hashtags_filter_columns(df)
        embeddings_filtered = filter_embeddings(df, embeddings, args.parties)
        save_filtered_embeddings(embeddings_filtered, args.model, args.year, "no-hashtags")

