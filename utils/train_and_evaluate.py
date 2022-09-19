import torch
import time
import torch.nn.functional as F

def train_epoch(model, optimizer, data_loader, device, loss_fun, images_per_class, val_images_per_class, num_classes, mixup):
    
    model.train()
    total_loss = 0

    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        (data, target) = mixup(data, target)     

        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
            
    avg_loss = total_loss /((images_per_class-val_images_per_class)*num_classes)
    print('Average train loss: ' + '{:.5f}'.format(avg_loss))
    return avg_loss


def evaluate_epoch(model, data_loader, device, loss_fun, val_images_per_class, num_classes):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in data_loader:

            data = data.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = loss_fun(output, target)
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / (val_images_per_class*num_classes)
    acc = 100*(correct_samples / total_samples)

    print('Average val loss: ' + '{:.5f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)')

    return avg_loss, acc