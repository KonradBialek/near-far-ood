from typing import List
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision
from torchsummary import summary

BATCH_SIZE = 256


def runTensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs'])
    url = tb.launch()
    print(f"TensorBoard on {url}")
    global writer
    writer = SummaryWriter()


def dataloader(dataset_path: str, normalization: List[List[float]]):
    global trainloader, testloader

    transform = transforms.Compose([
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(size[0]),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(normalization),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(size[0]),
        transforms.ToTensor(),
        transforms.Normalize(normalization),
        ])

    trainset = torchvision.datasets.ImageFolder(dataset_path + 'train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(dataset_path + 'test', transform=transform_val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def isCuda():
    global use_gpu
    use_gpu = torch.cuda.is_available()
    print('GPU: '+use_gpu)

def validate(model, criterion, epoch):
    loss_ = AverageMeter()
    model.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(testloader):
        if use_gpu: 
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss_.update(loss.item())

    acc = 100. * correct / total
    print(f"Validate: Loss: {loss_.val:.8f} ({loss_.avg:.8f}), Accuracy: {acc:.8f}")
    if epoch >= 0:
        writer.add_scalar("loss/val", loss_.avg, epoch)
        writer.add_scalar("acc/val", acc, epoch)
    
    return loss_.avg



# Train the model
def train_(model, criterion, optimizer, epoch):
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

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = 100. * correct / total


        if (i+1) % 100 == 0:
            print (f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss_:.4f}, Accuracy: {acc:.4f}")

    print (f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss_[0].val:.8f} ({loss_[0].avg:.8f}), Accuracy: {acc:.8f}")

    if epoch >= 0:
        writer.add_scalar("loss/train", loss_.avg, epoch)
        writer.add_scalar("acc/train", acc, epoch)



def train(nn: str, dataset: str, checkpoint: str):
    global checkpoints, best_loss, best_epoch, patience, epochs, size, use_gpu
    checkpoints = 'checkpoints'
    best_loss = torch.tensor(1e10)
    patience = 50
    epoch, best_epoch, epochs = 0, -1, 1000

    dataset_path = f'../data/{dataset}/'
    if dataset == 'cifar10': # todo generalize
        shape = cv2.imread(f'{dataset_path}test/airplane/0001.png').shape
        normalization = [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768] # mean, std
    elif dataset == 'mnist':
        shape = cv2.imread(f'{dataset_path}test/0_3.jpg').shape
        normalization = [0.13062754273414612], [0.30810779333114624] # mean, std
    size = shape[:1]

    os.makedirs(checkpoints, exist_ok=True)
    isCuda()
    runTensorboard()
    dataloader(dataset_path, normalization)
    model = torch.hub.load('pytorch/vision:v0.10.0', nn) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()


    if use_gpu: 
        criterion = criterion.cuda()
        model = model.cuda()
        summary(model, shape)    
        
    if checkpoint != '' and checkpoint is not None:
        path = f'./checkpoints/{checkpoint}'
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        model.train()


    for epoch in range(epoch, epochs):
        train_(model, criterion, optimizer, epoch)
        with torch.no_grad():
            loss = validate(model, criterion, epoch)

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'{checkpoints}/epoch-{epoch}-MSELoss-{loss:.8f}.pth')
        
        if epoch - best_epoch >= patience and epoch >= 100:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'{checkpoints}/epoch-{epoch}-MSELoss-{loss:.8f}-early_stop.pth')
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, f'{checkpoints}/epoch-{epoch}-last-{loss:.8f}.pth')

    with torch.no_grad():
        validate(model, criterion, -1)


def measure(nn, methd, in_dataset, out_datasets, checkpoint):
    in_dataset_path = f'../data/{in_dataset}/'
    out_datasets = out_datasets.split(' ')
    out_datasets_paths = [f'../data/{out_dataset}/' for out_dataset in out_datasets]

    model = torch.hub.load('pytorch/vision:v0.10.0', nn) 
    isCuda()
    if use_gpu:
        model = model.cuda()

    path = f'./checkpoints/{checkpoint}'
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()


    raise NotImplementedError()

