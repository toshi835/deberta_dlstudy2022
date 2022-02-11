import torch
from transformers import DebertaModel
from sklearn.metrics import accuracy_score


class DebertaClass(torch.nn.Module):
    def __init__(self, model_name, output_size, drop_rate=0.2):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(768, int(output_size))

    def forward(self, ids, mask):
        outputs = self.deberta(ids, attention_mask=mask)
        encoder_layer = outputs.last_hidden_state[:, 0]
        droped_output = self.dropout(encoder_layer)
        logits = self.classifier(droped_output)
        return logits


def calculate_loss_and_accuracy(model, criterion, loader, gpus):
    model.eval()
    loss = 0.0
    golds = []
    preds = []
    with torch.no_grad():
        for data in loader:
            # set gpu device
            ids = data['ids'].cuda(gpus)
            mask = data['mask'].cuda(gpus)
            labels = data['labels'].cuda(gpus)

            # forward
            outputs = model.forward(ids, mask)

            # calculate loss
            loss += criterion(outputs, labels).item()

            # calculate acculacy
            pred = torch.argmax(outputs, dim=-1).cpu().tolist()
            preds.extend(pred)
            golds.extend(labels.cpu().tolist())

    return loss / len(loader), accuracy_score(preds, golds)
