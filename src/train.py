from sklearn.utils import shuffle
import torch
import argparse
from tqdm import tqdm

from transformers import DebertaTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import CreateDataset
from model import DebertaClass, calculate_loss_and_accuracy


def main(args):
    num_class = 2

    # Preprocessing
    tokenizer = DebertaTokenizer.from_pretrained(args.model_name)
    # load dataset
    dataset = CreateDataset(args.data_path, tokenizer)
    # split data to train and devlop
    dataset_train, dataset_dev = train_test_split(
        dataset, test_size=args.dev_size, shuffle=True)
    print('Train data size: {}'.format(len(dataset_train)))
    print('Development data size: {}'.format(len(dataset_dev)))
    train_loader = DataLoader(dataset=dataset_train, batch_size=16)
    dev_loader = DataLoader(dataset=dataset_dev, batch_size=16)

    # creating model
    model = DebertaClass(args.model_name, num_class)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    if not torch.cuda.is_available():
        print('Could not find gpu resorces')
        assert False

    print('Start training')
    for epoch in range(args.epoch):
        loss_train = 0

        model = model.cuda(args.gpus)
        model.train()
        for data in tqdm(train_loader):
            # set gpu devices
            ids = data['ids'].cuda(args.gpus)
            mask = data['mask'].cuda(args.gpus)
            labels = data['labels'].cuda(args.gpus)

            # forward
            optimizer.zero_grad()
            outputs = model.forward(ids, mask)

            # calculate loss and update
            loss = criterion(outputs, labels)

            loss_train += loss.item()
            loss.backward()
            optimizer.step()

        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, criterion, dev_loader, args.gpus)

        print('Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(
            epoch + 1, loss_train/len(train_loader), loss_valid, acc_valid))

        print(
            f"Saving the model to '{args.out_path}_checkpoint{epoch + 1}.pt'")
        torch.save(model.to('cpu').state_dict(),
                   f'{args.out_path}_checkpoint{epoch + 1}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpus', default=0, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-epoch', default=10, type=int)
    parser.add_argument('-dev_size', default=872, type=int)
    parser.add_argument(
        '-data_path', default='../data/SST-2/train.tsv', type=str)
    parser.add_argument('-out_path', default='../model/sst2', type=str)
    parser.add_argument(
        '-model_name', default='microsoft/deberta-base', type=str)
    args = parser.parse_args()
    main(args)
