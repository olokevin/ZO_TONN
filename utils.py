from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_layers.layers import TensorizedLinear

def get_kl_loss(model, args, epoch):

    kl_loss = 0.0
    for layer in model.modules():
        if hasattr(layer, "tensor"):

            kl_loss += layer.tensor.get_kl_divergence_to_prior()
    kl_mult = args.kl_multiplier * torch.clamp(
                            torch.tensor((
                                (epoch - args.no_kl_epochs) / args.warmup_epochs)), 0.0, 1.0)
    """
    print("KL loss ",kl_loss.item())
    print("KL Mult ",kl_mult.item())
    """
    return kl_loss*kl_mult.to(kl_loss.device)



def get_net(args):
    if args.model_type in ['CP','TensorTrain','TensorTrainMatrix','Tucker']:
        return get_TensorizedNet(args)
    else:
        return Net()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        if args.rank_loss:
            ard_loss = get_kl_loss(model,args,epoch)
            loss += ard_loss
        


        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_TensorizedNet(args):

    if args.model_type=='full':
        fc1 = nn.Linear(784, 512)
        fc2 = nn.Linear(512, 10)
     
    else:
        if args.model_type=='TensorTrainMatrix':
            shape1 = [[4,7,7,4], [4,8,8,4]]   
            shape2 = [[4,8,8,4], [1,5,2,1]]     

        else:
            shape1 = [28, 28, 32, 32]
            shape2 = [32, 32, 10]

    
        fc1 = TensorizedLinear(784, 1024, shape=shape1, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
        fc2 = TensorizedLinear(1024, 10, shape=shape2, tensor_type=args.model_type,max_rank=args.rank,em_stepsize=args.em_stepsize)
    
    return TensorizedNet(fc1,fc2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784, 512, bias=False)
        self.fc2 = nn.Linear(512, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TensorizedNet(nn.Module):
    def __init__(self,fc1,fc2):
        super(TensorizedNet, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.add_module('fc1',fc1)
        self.add_module('fc2',fc2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class OrthoTONN(nn.Module):
  def __init__(self,tensor_type,max_rank,dropouts=0.5,prior_type='log_uniform',eta=1.0,device=None,dtype=None):
    super(OrthoTONN, self).__init__()
    self.dropout = nn.Dropout(dropouts)
    self.shape1 = [[4,7,7,4], [4,8,8,4]]   
    self.shape2 = [[4,8,8,4], [1,5,2,1]]     
    self.fc1 = TensorizedLinear(784, 1024, bias=None, shape=self.shape1, tensor_type=tensor_type, max_rank=max_rank,
                                prior_type=prior_type, eta=eta, device=device, dtype=dtype)
    self.fc2 = TensorizedLinear(1024, 10, bias=None, shape=self.shape2, tensor_type=tensor_type, max_rank=max_rank,
                                prior_type=prior_type, eta=eta, device=device, dtype=dtype)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = torch.flatten(x,1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

'''
  AddGaussianNoise to input data
'''
class AddGaussianNoise(object):
  def __init__(self, mean=0., std=1.):
    self.std = std
    self.mean = mean
      
  def __call__(self, tensor):
    return tensor + torch.randn(tensor.size()) * self.std + self.mean
  
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    