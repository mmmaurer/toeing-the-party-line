import fasttext
import transformers
from sentence_transformers import SentenceTransformer 
import polars as pl
import torch
import numpy as np
import pickle
from tqdm import tqdm

from .data_processing import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fasttext_embeddings(sentences, modelname="./cc.de.300.bin"):
    # account for potential linebreaks in the tweets
    # fasttext cannot handle linebreaks
    sentences = [sent.replace("\n", " ") for sent in sentences]
    # fasttext.util.download_model('de', if_exists='ignore')
    model = fasttext.load_model(modelname)
    embeddings = [model.get_sentence_vector(sent) for sent in sentences]
    return embeddings

def bert_embeddings(sentences, modelname, batch_size=64):
    model = transformers.AutoModel.from_pretrained(modelname,
                                                   cache_dir="./cache/"). \
                                                    to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelname, 
                                                           cache_dir="./cache/")
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt",
                               padding=True,
                               truncation=True).to(device)
            output = model(**inputs)
            batch_embeddings = output.pooler_output.cpu().detach().numpy()
            embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def sbert_embeddings(sentences, modelname, show_progress_bar=False, device="cuda"):
    model = SentenceTransformer(modelname, cache_folder="./cache/", 
                                device=device)
    embeddings = model.encode(sentences,
                              convert_to_tensor=True,
                              show_progress_bar=show_progress_bar). \
                                detach().cpu().numpy()
    return embeddings

def get_embeddings(path, modelname, output_dir, year=2013,
                   parties=['AfD', 'CDU/CSU', 'FDP',
                            'GRÜNE', 'DIE LINKE', 'SPD']):
    df = load_data(path, year)
    representations_per_party = []
    for party in tqdm(parties):
        sentences = df.filter(pl.col("faction")==party)["text"].to_list()
        if "fasttext" in modelname:
            embeddings = fasttext_embeddings(sentences)
        elif "bert" in modelname:
            embeddings = bert_embeddings(sentences, modelname)
        else:
            embeddings = sbert_embeddings(sentences, modelname)
        representations_per_party.append(embeddings)
    if modelname=="./out/":
        pickle.dump(representations_per_party, open(output_dir + "/" + \
                                                    f"sbert-ht-{year}.pkl", "wb"))
    else:
        pickle.dump(representations_per_party, open(output_dir + "/" + \
                                                    f"{modelname}-{year}.pkl", "wb"))

