from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import torch
import argparse

def load_data(path, year, is_eval=False):
    """Load the data from the dataset.
    Assumes the dataset is in the format of a csv file with columns
    "sent1", "sent2", "label" for the training set and "sent1", "sent2",
    "label" for the evaluation set.

    
    Args:
    - path: Path to the dataset, should contain the csv files
            {train, val}_{year}.csv
    - year: Year of the dataset to load
    - is_eval: Boolean, whether to load the evaluation set or not

    Returns:
    - examples: List of InputExamples
    """
    examples = []
    if not is_eval:
        df = pd.read_csv(f"{path}/train_{year}.csv", lineterminator="\n")
    else:
        df = pd.read_csv(f"{path}/val_{year}.csv", lineterminator="\n")
    
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[row["sent1"],
                                            row["sent2"]],
                                     label=row["label"]))
    
    return examples


def train(data_path,
          output_path,
          year,
          batch_size=16,
          epochs=5,
          evaluation_steps=100,
          warmup_steps=100,
          pool=None,
          device='cuda:0'
          ):
    """
    Train a SentenceTransformer model on the given dataset

    Args:
    - data_path: Path to the dataset dir, should contain a train.csv
                 and an eval.csv
    - output_path: Path to the output dir
    - year: Year of the dataset to use
    - batch_size: Batch size for training
    - epochs: Number of epochs to train for
    - evaluation_steps: Number of steps between evaluations
    - warmup_steps: Number of warmup steps
    - pool: GPU pool to use for training
    - device: Device to use for training
    
    Returns:
    - None
    """
    train_examples = load_data(data_path, year)

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)

    if pool is not None:
        devices = []
        for gpu in pool.split("-"):
            devices.append(f"cuda:{gpu}")
        model.start_multi_process_pool(target_devices=devices)

    train_dataloader = DataLoader(train_examples, shuffle=True, 
                                  batch_size=batch_size)
    train_loss = losses.ContrastiveLoss(model)

    eval_examples = load_data(data_path, year, is_eval=True)
    evaluator = evaluation.EmbeddingSimilarityEvaluator. \
                           from_input_examples(eval_examples, name="eval")
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=epochs,
              warmup_steps=warmup_steps,
              evaluation_steps=evaluation_steps,
              output_path=output_path,
              save_best_model=True
              )

def evaluate(data_path, year, output_path, device='cuda:0'):
    """
    Evaluate a SentenceTransformer model on the given dataset

    Args:
    - data_path: Path to the dataset dir, should contain a train.csv
                 and an eval.csv
    - year: Year of the dataset to use
    - output_path: Path to the output dir
    - device: Device to use for evaluation

    Returns:
    - None
    """
    model = SentenceTransformer(output_path, device=device)
    eval_examples = load_data(data_path, year, is_eval=True)

    evaluator = evaluation.EmbeddingSimilarityEvaluator. \
                           from_input_examples(eval_examples, name="eval")
    
    evaluator(model, output_path=output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the dataset dir,"
                        "should contain a train.csv and an eval.csv")
    parser.add_argument("--year", help="Year of the dataset to use")
    parser.add_argument("--output_path", help="Path to output dir")
    parser.add_argument('--batch_size', nargs="?", type=int, default=32)
    parser.add_argument('--epochs', nargs="?", type=int, default=5)
    parser.add_argument('--warmup_steps', nargs="?", type=int, default=1000)
    parser.add_argument('--evaluation_steps', nargs="?", type=int,
                        default=100)
    parser.add_argument('--pool', nargs="?", type=str, default=None)
    parser.add_argument('--cuda', nargs="?", type=str, default='0')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}")
    train(args.data_path, args.output_path, args.year, args.batch_size, args.epochs,
          args.evaluation_steps, args.warmup_steps, args.pool, device=device)
    
    evaluate(args.data_path, args.year, args.output_path, device=device)

