import numpy as np
from sentence_transformers import util
from itertools import combinations, product

def hashtag_distance_matrix(embeddings_per_hashtag, parties):
    """Computes a hashtag distance matrix.
    Args:
        embeddings_per_hashtag: Dictionary of hashtags mapped to a list of
                                embeddings per party.
                                Structure:
                                {   
                                    '#hashtag':{
                                        party_abbreviation:embeddings_array,
                                    }
                                }
        parties: list of party abbreviations

    Returns:
        distance_matrix: Hashtag decomposition matrix
    """
    distance_matrices = np.array()
    # Distance per hashtag
    for tag in embeddings_per_hashtag.keys():
        distance_matrix = np.zeros((len(parties),len(parties)))
        for i,j in combinations([n for n in range(len(parties))], 2):
            dist_ij = category_distance(embeddings_per_hashtag[tag][parties[i]],
                                        embeddings_per_hashtag[tag][parties[j]])
            distance_matrix[i, j] = dist_ij
            distance_matrix[j, i] = dist_ij

        np.append(distance_matrices, distance_matrix)
    # Returning overall matrix
    distance_matrix_overall = np.mean(distance_matrices, axis=0)
    return distance_matrix_overall

def category_distance(embeddings_a, embeddings_b):
    """Distance per category, as defined by Ceron et al (2023).

    Args:
        embeddings_a: Array of the embeddings of texts for a given category
                      from party a
        embeddings_b: Array of the embeddings of texts for a given category
                      from party b
    
    Returns:
        distance_ab: Distance between the two parties ito the given category
    """
    # Distance per sentence pair, resulting shape:
    # (# of sentences a, # of sentences b)
    cosine_similarities = util.cos_sim(np.array(embeddings_a),
                            np.array(embeddings_b)).numpy()
    # Sum over 1 - cos(), equivalent matrix form below
    distance_ab = np.mean(np.sum(np.ones((cosine_similarities.shape[0],
                                          cosine_similarities.shape[1])) - \
                                            cosine_similarities))
    return distance_ab
    

def twin_matching_similarity(embeddings_a, embeddings_b):
    """Twin matching similarity definition as proposed by Ceron et
    al. (2022). Note that sim(a,b) != sim(b,a) and sim(a,a) != 1.

    Args:
        embeddings_a: Array of the embeddings of texts from party a
        embeddings_b: Array of the embeddings of texts from party b

    Returns:
        similarity_ab: Similarity score between the two parties
    """
    cosine_similarities_a = util.cos_sim(np.array(embeddings_a),
                                         np.array(embeddings_a)).numpy()
    cosine_similarities_b = util.cos_sim(np.array(embeddings_b),
                                         np.array(embeddings_b)).numpy()
    cosine_similarities_ab = util.cos_sim(np.array(embeddings_a),
                          np.array(embeddings_b)).numpy()
    
    # Setting diagonal to 0 to avoid using the identical tweet
    # as intra-set twin; with that cosine_similarities_a and
    # cosine_similarities_b would always be 1
    np.fill_diagonal(cosine_similarities_a, 0)
    np.fill_diagonal(cosine_similarities_b, 0)

    C_a = np.max(cosine_similarities_a, axis=1)
    C_b = np.max(cosine_similarities_b, axis=1)

    arg_twins_a = np.argmax(cosine_similarities_ab, axis=1)

    similarity_ab = (1/len(cosine_similarities_a)) * np.sum([
        cosine_similarities_ab[idx_a][idx_b]/(C_a[idx_a]+C_b[idx_b])
        for idx_a, idx_b in enumerate(arg_twins_a)])
    return similarity_ab

def twin_matching_similarity_matrix(embeddings):
    """Computes a twin matching similarity matrix for a set of parties.

    Args:
        embeddings: Array with an array of embeddings per party

    Returns:
        similarity matrix: Twin matching similarity matrix 
    """
    # Embedding array should have length num_of_parties
    similarity_matrix = np.zeros((len(embeddings),len(embeddings)))
    permutations = [(i,j) for (i,j) in product(range(len(embeddings)),
                                               repeat=2)]
    for (i,j) in permutations:
        ij = twin_matching_similarity(embeddings[i], embeddings[j])
        similarity_matrix[i][j] = ij
        del ij  # for memory reasons
    return similarity_matrix

def twin_matching_distance_matrix(embeddings):
    """Computes a twin matching similarity matrix for a set of parties.

    Args:
        embeddings: Array with an array of embeddings per party

    Returns:
        similarity matrix: Twin matching similarity matrix 
    """
    similarity_matrix = twin_matching_similarity_matrix(embeddings)
    # diagonalizing
    np.fill_diagonal(similarity_matrix, 1)
    # transform into distance matrix
    distance_matrix = np.full(similarity_matrix.shape, 1) - \
        (similarity_matrix + similarity_matrix.T) / 2
    return distance_matrix
