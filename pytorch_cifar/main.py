'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import *
from mnist_utils import *
from time import time

from swiftclient.service import SwiftService, SwiftError
from swiftclient.exceptions import ClientException

def download_dataset(swift, dataset_name, datadir):
  """
  download dataset from swift db to local disk
	Parameters:
	swift                   Swift Service object to help download the dataset
        dataset_name (str)	name of the dataset to be downloaded (also the name of the container in swift db)
	datadir (str)		local directory on which the dataset should be stored
  """
  page_list = swift.list(container=dataset_name)
  opts = {'out_directory':datadir}
  for page in page_list:
    if page['success']:		#Otherwise, there is an error...skip
      objects = [obj['name'] for obj in page['listing']]
      for res in swift.download(container=dataset_name, objects=objects, options=opts):
        if not res['success']:
          print("ERROR!!", res)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--testonly', action='store_true', help='run inference only without training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset to be used')
parser.add_argument('--model', default='convnet', type=str, help='model to be used')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for dataloader')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs for training')
args = parser.parse_args()

dataset_name = args.dataset
model = args.model
task = args.task
batch_size = args.batch_size
num_epochs = args.num_epochs
print(args)

start_time = time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

homedir = os.environ['HOME']
datadir = os.path.join(homedir,"dataset",dataset_name)
#first fetch data....we assume here that data does not exist locally
swift = SwiftService()
start_download_t = time()
try:
  download_dataset(swift, dataset_name, datadir)
except ClientException as e:
  print("Got an exeption while downloading the dataset ", e)
print('data downloaded...time elapsed: {}'.format(time()-start_download_t))

if dataset_name.startswith('cifar'):
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  trainset = torchvision.datasets.CIFAR10(
    root=datadir, train=True, download=False, transform=transform_train)
  testset = torchvision.datasets.CIFAR10(
    root=datadir, train=False, download=False, transform=transform_test)
elif dataset_name == 'mnist':
  transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, ))
              ])
###########################################################################################################
#This is the work that should be done to prepare the MNIST dataset...the same piece of code is excted in native PyTorch code
  training_set = (
  	read_image_file(os.path.join(datadir,'mnist', 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(datadir,'mnist', 'train-labels-idx1-ubyte'))
  )
  test_set = (
        read_image_file(os.path.join(datadir, 'mnist', 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(datadir, 'mnist', 't10k-labels-idx1-ubyte'))
  )
  processed_folder = 'processed'
  training_file = 'training.pt'
  test_file = 'test.pt'
  os.makedirs(processed_folder, exist_ok=True)
  with open(os.path.join(datadir, processed_folder, training_file), 'wb') as f:
    torch.save(training_set, f)
  with open(os.path.join(datadir, processed_folder, test_file), 'wb') as f:
    torch.save(test_set, f)
###########################################################################################################
  trainset = datasets.MNIST(root=datadir, train=True, download=False, transform=transform)
  testset = datasets.MNIST(root=datadir, train=False, download=False, transform=transform)
elif dataset_name == 'iamgenet':
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
  ])
  transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
  ])
  #First, we need to put images correctly in its folders
  os.system("cd {}; ./valprep.sh".format(os.path.join(datadir,'imagenet')))
  #Then, we load the Imagenet dataset
  trainset = datasets.ImageFolder(root=datadir, transform=transform_train)
  testset = datasets.ImageFolder(root=datadir, transform=transform_test)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = get_model(model, dataset_name)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

#To maintain fairness, we should not do extra computation(s)
#        train_loss += loss.item()
#        _, predicted = outputs.max(1)
#        total += targets.size(0)
#        correct += predicted.eq(targets).sum().item()

#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    res = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            res.extend(predicted)
#            total += targets.size(0)
#            correct += predicted.eq(targets).sum().item()

#            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
#    acc = 100.*correct/total
#    if acc > best_acc:
#        print('Saving..')
#        state = {
#            'net': net.state_dict(),
#            'acc': acc,
#            'epoch': epoch,
#        }
#        if not os.path.isdir('checkpoint'):
#            os.mkdir('checkpoint')
#        torch.save(state, './checkpoint/ckpt.pth')
#        best_acc = acc
    return res

if args.testonly:
    res = test(0)
    print("Inference done for {} inputs".format(len(res)))
    print("The whole process took {} seconds".format(time()-start_time))
    exit(0)

for epoch in range(start_epoch, num_epochs):
    train(epoch)
#    test(epoch)
    scheduler.step()
print("The whole process took {} seconds".format(time()-start_time))
