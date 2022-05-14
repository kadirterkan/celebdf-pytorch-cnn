import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograf import Variable

from model.tripletnet import get_model
from dataset.triplet_img_loader import get_triplet_dataset

#COPIED FROM https://github.com/avilash/pytorch-siamese-triplet

def train(data, model, criterion, optimizer, epoch):
    print("******** Training ********")
    total_loss = 0
    model.train()
    for batch_idx, img_triplet in enumerate(data):
        anchor_img, pos_img, neg_img = img_triplet
        anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_step = train_log_step
        if (batch_idx % log_step == 0) and (batch_idx != 0):
            print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
            total_loss = 0
    print("****************")


def test(data, model, criterion):
    print("******** Testing ********")
    with torch.no_grad():
        model.eval()
        accuracies = [0, 0, 0]
        acc_threshes = [0, 0.2, 0.5]
        total_loss = 0
        for batch_idx, img_triplet in enumerate(data):
            anchor_img, pos_img, neg_img = img_triplet
            anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
            anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
            E1, E2, E3 = model(anchor_img, pos_img, neg_img)
            dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
            dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

            target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            target = target.to(device)
            target = Variable(target)

            loss = criterion(dist_E1_E2, dist_E1_E3, target)
            total_loss += loss

            for i in range(len(accuracies)):
                prediction = (dist_E1_E3 - dist_E1_E2 - margin * acc_threshes[i]).cpu().data
                prediction = prediction.view(prediction.numel())
                prediction = (prediction > 0).float()
                batch_acc = prediction.sum() * 1.0 / prediction.numel()
                accuracies[i] += batch_acc
        print('Test Loss: {}'.format(total_loss / len(data)))
        for i in range(len(accuracies)):
            print(
                'Test Accuracy with diff = {}% of margin: {}'.format(acc_threshes[i] * 100, accuracies[i] / len(data)))
    print("****************")

def main():
    global device, train_log_step, margin

    train_log_step = 100
    margin = 1.0

    epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model('vggface2', device)

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value]}]
    criterion = torch.nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(params, lr=0.0001)

    for epoch in range(1, epochs + 1):
        # Init data loaders
        train_data_loader, test_data_loader = get_triplet_dataset()
        # Test train
        test(test_data_loader, model, criterion)
        train(train_data_loader, model, criterion, optimizer, epoch)
        # Save model
        model_to_save = {
            "epoch": epoch + 1,
            'state_dict': model.state_dict(),
        }



