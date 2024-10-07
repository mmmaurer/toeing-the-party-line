import argparse
import os

from src.get_embeddings import get_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data",
                        help="Path to the data directory.",
                        type=str)
    parser.add_argument("--output_dir", default="embeddings",
                        help="Path to the output directory.",
                        type=str)
    parser.add_argument("--model", default="mmmaurer/sbert-hashtag-german-politicians-2021",
                        help="Model name to gather embeddings for.",
                        type=str)
    parser.add_argument("--year", default=2021, help="Year of the data.",
                        type=int)
    parser.add_argument("--cache_dir", default="./cache/",
                        help="Path to the Transformers cache directory.",
                        type=str)
    args = parser.parse_args()
    output_dir = args.output_dir
    data_path = args.data_dir
    model = args.model
    year = args.year

    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    
    print(f"Getting embeddings for {model}.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    get_embeddings(path=f"{data_path}/btw{year}_tweets.csv",
                   modelname=model, output_dir=output_dir,year=year)
    print("Done.")

if __name__ == "__main__":
    main()