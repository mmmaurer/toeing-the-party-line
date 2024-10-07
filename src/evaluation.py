import numpy as np
from itertools import combinations_with_replacement
import pandas as pd


PARTY_NAME_DICT = {"90/Greens":"GRÜNE",
                   "LINKE":"DIE LINKE",
                   "SPD":"SPD",
                   "FDP":"FDP",
                   "CDU/CSU":"CDU/CSU",
                   "AfD":"AfD"}

def manifesto_distance_matrix(path,
                              year,
                              parties,
                              mode="full",
                              countryname="Germany",
                              party_name_dict=PARTY_NAME_DICT):
    """Computes a ground truth distance matrix based on the MARPOR
    manifesto annotations. 
    Builds vectors out of relative salience of sentences per category
    and computes pairwise euclidean distance.

    modes: 
        - 'full' for all categories and subcategories
        - 'cat'  to only use proper categories (e.g. no per602_1)
        - 'rile' to only use rile categories
        - 'econ' to only use economic left-right scale categories
        - 'galtan' to only use galtan categories

    Args:
        path: path to the manifesto project csv
        year: Year in format YYYY
        parties: list of party abbreviations
        mode: the mode of matrix to compute

    Returns:
        mat: A numpy distance matrix
    """
    rile_cats = ["per104","per201","per203","per305","per401",
                 "per402","per407","per414","per505","per601",
                 "per603","per605","per606","per103","per105",
                 "per106","per107","per202","per403","per404",
                 "per406","per412","per413","per504","per506",
                 "per701"
    ]
    # Categories as defined in Bakker & Hobolt (2013)
    econ_cats = ["per401","per402","per407","per505","per410",
                 "per414","per702","per403","per404","per406",
                 "per504","per506","per413","per412","per701",
                 "per409","per415","per503"
    ]
    galtan_cats = ["per305","per601","per603","per605","per608",
                   "per606","per501","per602","per604","per502",
                   "per607","per416","per705","per706","per201",
                   "per202"
    ]

    year = str(year)  # making sure the year is a string
    df = pd.read_csv(path)
    # filtering only for relevant year and country to only
    # compute following steps for necessary rows
    df = df[(df.date.astype(str).str
             .startswith(year)) & (df.countryname==countryname)]
    df.partyabbrev = df.partyabbrev.replace(party_name_dict)
    df = df.fillna(0)  # to avoid problems processing the CMP data
    # columns starting with 'per' are fractions of sentences in policy
    # categories
    if mode=='full':
        cols = [col for col in df.columns if str(col).startswith("per")]
        # the following two are not categories
        cols.remove("pervote")
        cols.remove("peruncod")
    elif mode == 'cat':
        # proper category names have the structure perXXX -> len = 6 
        cols = [col for col in df.columns if \
                (str(col).startswith("per") and len(col)==6)]
    elif mode == 'rile':
        cols = [col for col in df.columns if col in rile_cats]
    elif mode == 'econ':
        cols = [col for col in df.columns if col in econ_cats]
    elif mode == 'galtan':
        cols = [col for col in df.columns if col in galtan_cats]
    else:
        print("Not a valid mode.")
        exit()

    vecs = {party:df[(df.partyabbrev.isin(parties))]
            .set_index("partyabbrev")[cols].T[party]
            .to_list()[1:] for party in parties}
    mat = np.zeros((len(parties),len(parties)))
    for (i, j) in combinations_with_replacement(range(len(parties)), 2):
        # Euclidean distance between parties
        mat[i,j] = np.linalg.norm(np.array(vecs[parties[i]]) - \
                                  np.array(vecs[parties[j]]))
        mat[j,i] = mat[i,j]
    return mat

