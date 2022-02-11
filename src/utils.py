import torch


class CreateDataset:
    def __init__(self, file_path, tokenizer, max_len=256, test=False):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.x = []
        self.y = []
        self.test = test

        print('Loading from {}'.format(self.file_path))
        self._build()
        print('Loading finished')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.test:
            return {
                'text': self.x[index]['src_text'],
                'ids': self.x[index]['input_ids'],
                'mask': self.x[index]['attention_mask'],
                'labels': self.y[index]
            }
        else:
            return {
                'ids': self.x[index]['input_ids'],
                'mask': self.x[index]['attention_mask'],
                'labels': self.y[index]
            }

    def _build(self):
        with open(self.file_path, 'r') as data:
            for i, d in enumerate(data):
                if i == 0:  # ignore index
                    continue
                src_text, target = d.strip().split('\t')
                src = self.tokenizer.encode_plus(
                    src_text, truncation=True, max_length=self.max_len, padding='max_length')
                if self.test:
                    inp = {'src_text': src_text, 'input_ids': torch.tensor(
                        src['input_ids']), 'attention_mask': torch.tensor(src['attention_mask'])}
                else:
                    inp = {'input_ids': torch.tensor(
                        src['input_ids']), 'attention_mask': torch.tensor(src['attention_mask'])}

                self.x.append(inp)
                self.y.append(int(target))
