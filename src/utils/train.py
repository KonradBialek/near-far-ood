import os
import torch
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from torch_optimizer import Lookahead

from .utils import getNN, getShapeNormalization, runTensorboard, dataloader, AverageMeter, isCuda, saveModel, showLayers, updateWriter

def validate_(model, valloader, testloader, criterion, epoch, use_gpu: bool):
    with torch.no_grad():
        if testloader is None:
            return validate(model, valloader, criterion, epoch, use_gpu)
        else:
            return validate(model, testloader, criterion, epoch, use_gpu)


def validate(model, loader, criterion, epoch: int, use_gpu: bool):
    loss_ = AverageMeter()
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        if use_gpu: 
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss_.update(loss.item())

    acc = 100. * correct / total
    print(f"Validate: Loss: {loss_.val:.8f} ({loss_.avg:.8f}), Accuracy: {acc:.8f}")
    if epoch >= 0:
        updateWriter("val", loss_.avg, acc, epoch)
    
    return loss_.avg


def train_(model, trainloader: torch.utils.data.DataLoader, criterion, optimizer, epoch: int, epochs: int, use_gpu: bool):
    loss_ = AverageMeter()
    model.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):
        if use_gpu: 
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        loss_.update(loss.item())

        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


        if (i+1) % 100 == 0:
            acc = 100. * correct / total
            print (f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss_.val:.8f} ({loss_.avg:.8f}), Accuracy: {acc:.8f}")

    acc = 100. * correct / total
    print (f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss_.val:.8f} ({loss_.avg:.8f}), Accuracy: {acc:.8f}")
    
    updateWriter("train", loss_.avg, acc, epoch)



# def train(nn: str, dataset: str, checkpoint: str, la_steps: int, la_alpha: float, n_holes: int, length: int):
def train(nn: str, dataset: str, checkpoint: str, n_holes: int, length: int, la_steps: int, la_alpha: float):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    checkpoints = f'checkpoints/{nn}/{dataset}/{date_time}'
    best_loss = torch.tensor(1e10)
    patience = 70
    epoch, best_epoch, epochs = 0, -1, 1000

    shape, normalization = getShapeNormalization(dataset)
    os.makedirs(checkpoints, exist_ok=True)
    use_gpu = isCuda()
    runTensorboard()
    trainloader, valloader, testloader = dataloader(dataset, size=shape[:2], train=True, setup=None, n_holes=n_holes, length=length, normalization=normalization)

    model = getNN(nn, dataset)
    criterion = torch.nn.CrossEntropyLoss()

    if use_gpu: 
        criterion = criterion.cuda()
        model = model.cuda()
    
    showLayers(model, shape)    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer = Lookahead(optimizer, k=la_steps, alpha=la_alpha)
    scheduler = MultiStepLR(optimizer, milestones=[20+40*x for x in range(1, 25)], gamma=0.1)

    if checkpoint != '' and checkpoint is not None:
        path = f'./checkpoints/{checkpoint}'
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch = ckpt['epoch']
        model.train()

    for epoch in range(epoch, epochs):
        train_(model, trainloader, criterion, optimizer, epoch, epochs, use_gpu)
        loss = validate_(model, valloader, testloader, criterion, epoch, use_gpu)
        
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            saveModel(epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), loss, checkpoints, nn, 0)
        
        if epoch - best_epoch >= patience and epoch >= 100:
            saveModel(epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), loss, checkpoints, nn, 1)
            break

    saveModel(epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), loss, checkpoints, nn, 2)

    validate_(model, valloader, testloader, criterion, epoch, use_gpu)
