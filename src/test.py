import torch
import argparse

from tqdm import tqdm
from transformers import DebertaTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from utils import CreateDataset
from model import DebertaClass


def main(args):
    num_class = 2

    # Preprocessing
    tokenizer = DebertaTokenizer.from_pretrained(args.model_name)
    # load dateset
    dataset_test = CreateDataset(args.data_path, tokenizer, test=True)
    print('Test data size: {}'.format(len(dataset_test)))
    test_loader = DataLoader(dataset=dataset_test, batch_size=16)

    if not torch.cuda.is_available():
        print('Could not find gpu resorces')
        assert False

    # Create model
    model = DebertaClass(num_class)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda(args.gpus)

    model.eval()
    texts = []
    golds = []
    preds = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            # set gpu devices
            ids = data['ids'].cuda(args.gpus)
            mask = data['mask'].cuda(args.gpus)

            # Predict
            outputs = model.forward(ids, mask)
            pred = torch.argmax(outputs, dim=-1).cpu()

            texts.extend(data['text'])
            golds.extend(data['labels'].tolist())
            preds.extend(pred.tolist())

    # Result
    print('accuracy: {0:.4f}'.format(accuracy_score(golds, preds)))
    print('confusion matrix:')
    print(confusion_matrix(golds, preds))
    print()

    # Sample Result
    for i in range(10):
        print('Sentence, Gold label, Predict label')
        print('{}, {}, {}'.format(texts[i], golds[i], preds[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model_path', required=True, type=str)
    parser.add_argument('-gpus', default=0, type=int)
    parser.add_argument(
        '-data_path', default='../data/SST-2/test.tsv', type=str)
    parser.add_argument(
        '-model_name', default='microsoft/deberta-base', type=str)
    args = parser.parse_args()
    main(args)
