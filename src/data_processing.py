import polars as pl
import re

# These are the datatypes for the columns in the data
# used in our experiments. Depending on the dataset,
# these may need to be adjusted 
DTYPES = {
    'cand_id': pl.UInt16,
    'lastname_last': pl.String,
    'firstname_last': pl.String,
    'lastname_r_last': pl.String,
    'firstname_r_last': pl.String,
    'gender': pl.String,
    'birthyear': pl.UInt16,
    'cand_id_duplicated_0': pl.UInt16,
    'election_id': pl.String,
    'lastname': pl.String,
    'firstname': pl.String,
    'lastname_r': pl.String,
    'firstname_r': pl.String,
    'profession': pl.String,
    'party': pl.String,
    'incumbent': pl.UInt8,
    'elected': pl.UInt8,
    'cand_type': pl.String,
    'list_no': pl.UInt16,
    'list_state': pl.String,
    'district_no': pl.UInt16,
    'district_name': pl.String,
    'district_state': pl.String,
    'gles_id_13': pl.UInt16,
    'birthplace': pl.String,
    'residence': pl.String,
    'gles_id_17': pl.UInt16,
    'gles_id_21': pl.UInt16,
    'state': pl.String,
    'invited': pl.String,
    'participated': pl.String,
    'mode': pl.String,
    'ltw_id': pl.String,
    'district_2_no': pl.UInt16,
    'district_2_name': pl.String,
    'bezirk_name': pl.String,
    'nomination_type': pl.String,
    'party_long': pl.String,
    'cand_id_duplicated_1': pl.UInt16,
    'twitter_id': pl.String,
    'screen_name': pl.String,
    'tweet_id': pl.String,
    'twitter_id_duplicated_0': pl.String,
    'conversation_id': pl.String,
    'lang': pl.String,
    'is_retweet': pl.UInt8,
    'is_quoted': pl.UInt8,
    'is_reply': pl.UInt8,
    'text': pl.String,
    'likes': pl.UInt16,
    'image_url': pl.String,
    'retweets': pl.UInt16,
    'quotes': pl.UInt16,
    'replies': pl.UInt16,
    'created_at': pl.Datetime,
}

def preprocess_parties(df):
    """Preprocesses the parties in the dataframe
       such that CDU and CSU are unified as CDU/CSU
       as they are one faction in the Bundestag.
       To still access the individual parties, the
       original party name is kept in a column called
       'party'. The new column is called 'faction'.  
    
    Args:
        df: A polars dataframe
        
    Returns:
        df: The dataframe with the new column factions
    """
    return df.with_columns(
        [
            pl.when((pl.col("party") == "CSU") | (pl.col("party") == "CDU"))
            .then(pl.lit("CDU/CSU"))
            .otherwise(pl.col("party")).alias("faction"),
        ]
    )

def get_year(df):
    """Creates year column from a polars dataframe.
    The year is extracted from the 'created_at' column.
    Thus, it refers to the year the tweet was created in.
    
    Args:
        df: A polars dataframe

    Returns:
        df: The dataframe with a column 'year' containing the year
    """
    return df.with_columns(
        [
            pl.col("created_at").dt.year().alias("year"),
        ]
    )

def get_month(df):
    """Creates month column from a polars dataframe.
    The month is extracted from the 'created_at' column.
    Thus, it refers to the month the tweet was created in.
    
    Args:
        df: A polars dataframe

    Returns:
        df: The dataframe with a column 'month' containing the month
    """
    return df.with_columns(
        [
            pl.col("created_at").dt.month().alias("month"),
        ]
    )

def get_day(df):
    """Creates day column from a polars dataframe.
    The day is extracted from the 'created_at' column.
    Thus, it refers to the day the tweet was created on.
    
    Args:
        df: A polars dataframe

    Returns:
        df: The dataframe with a column 'day' containing the day
    """
    return df.with_columns(
        [
            pl.col("created_at").dt.day().alias("day"),
        ]
    )

def load_data(path, year):
    """Loads data from a csv for a specific year.
    The data is preprocessed such that the factions
    can be extracted. The year is also added as a column,
    and the data is filtered for the specific year.

    Args:
        path: The path to the csv file
        year: The year to filter for
    
    Returns:
        df: The preprocessed dataframe
    """
    df = pl.read_csv(path, has_header=True, separator=",",
                     ignore_errors=True, dtypes=DTYPES)
    df = get_year(df)
    df = df.filter(pl.col("year") == int(year))
    df = preprocess_parties(df)
    return df

def extract_hashtags(text):
    """Extracts hashtags from a string
    """
    return re.findall(r'#\w+', text)


def get_hashtags(df, filter_list=None):
    """Given a polars dataframe with tweets in a column called 'text',
    retrieves hashtags, cleans them up and appends a column called
    hashtags to the df.

    Args:
        df: The dataframe containing the tweets
        filter_list (optional): A list of hashtags (strings) to find
                                in the tweets

    Returns:
        df: The dataframe with a column called 'hashtags' containing
            the list of occurring hashtags appended
    """
    if filter_list is not None:
        filter_list = [f"#{tag}" for tag in filter_list]
        res = df.filter(pl.col("text").str.contains("|".join(filter_list)))
    else:
        res = df.with_columns(
            pl.col("text").apply(extract_hashtags).alias("hashtags")
        )
    return res

def get_representations_per_hashtag(embeddings, 
                                    df,
                                    parties):
    """Maps representations to hashtags that occur in the respective
    tweets.
    
    Args:
        embeddings: List of embeddings per party
        df: A dataframe containing hashtags in a column 'hashtags'
        parties: List of strings containing the party abbreviations

    Returns:
        tweet_emb_per_hashtags: A dict with a dict per hashtag mapping
                                parties to respective embeddings.
    """
    # members of parliament (mps) and restrictions (restr)
    # not implemented (yet)
    tweet_emb_per_hashtags = {}
    for i, party in enumerate(parties):
        hashtags_list = df.filter(pl.col("faction") == party)["hashtags"].to_list()
        for j, hashtags in enumerate(hashtags_list):
            for hashtag in hashtags:
                if hashtag not in tweet_emb_per_hashtags.keys():
                    tweet_emb_per_hashtags[hashtag] = {party:[embeddings[i][j]]}
                elif party not in tweet_emb_per_hashtags[hashtag] \
                    .keys():
                    tweet_emb_per_hashtags[hashtag][party] = \
                        [embeddings[i][j]]
                else:
                    tweet_emb_per_hashtags[hashtag][party] \
                        .append(embeddings[i][j])
    return tweet_emb_per_hashtags

def get_common_hashtags(df, num_parties=6, threshold=50):
    """Retrieves a list of common hashtags across parties given a data
    frame with hashtags column.

    Args:
        df: A dataframe with the hashtags in a list in a column called
            'hashtags'.
        num_parties: The number of parties any hashtag should at least
                     occur for.
        threshold: The number of tweets any hashtag should at least
                   appear in.
    Returns:
        common_hashtags: A list of hashtags
    """
    count_per_hashtag = {}
    parties_per_hashtag = {}

    party_per_tweet = df["faction"].to_list()
    hashtags_per_tweet = df["hashtags"].to_list()

    for i, hashtags in enumerate(hashtags_per_tweet):
        for hashtag in hashtags:
            if hashtag not in parties_per_hashtag.keys():
                parties_per_hashtag[hashtag] = {party_per_tweet[i]}
            else:
                parties_per_hashtag[hashtag].add(party_per_tweet[i])
            if hashtag not in count_per_hashtag.keys():
                count_per_hashtag[hashtag] = 1
            else:
                count_per_hashtag[hashtag] += 1
    common_hashtags = []
    for hashtag, parties in parties_per_hashtag.items():
        # Total number of parties, in the German data present here, 6
        if len(parties) >= num_parties \
            and count_per_hashtag[hashtag]>=threshold:
            common_hashtags.append(hashtag)
    return common_hashtags, count_per_hashtag, parties_per_hashtag

def hashtag_coocurrence(df, filter_list=None):
    """Finds coocurrence of hashtags in a polars dataframe containing tweets.
    Note that the column with the tweets has to be called 
    'tweet_text_data'
    
    Args:
        df: A polars dataframe containing tweets in a column 'tweet_text_data'
        filter_list: A list of hashtags (strings) to restrict to.

    Returns:
        cooc: A dict mapping hashtags to a list of other hashtags they
              cooccur with.
    """
    cooc = {}
    
    for taglist in df["hashtags"].to_list():
        if len(taglist) > 0:
            for tag in taglist:
                if tag in filter_list:
                    if tag not in cooc.keys():
                        cooc[tag] = [t for t in taglist if t!=tag]
                    else:
                        cooc[tag] += [t for t in taglist if t!=tag]
                    cooc[tag] += [tag]
    for tag in cooc.keys():
        cooc[tag] = set(cooc[tag])
    return cooc

