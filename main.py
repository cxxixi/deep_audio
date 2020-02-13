"""
 Learning from Between-class Examples for Deep Sound Recognition.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import torch
from utils import *
import models
from dataset import audio_esc50
from models.dnn import Net
import torch.optim as optim
import time
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler


def main():
    
    run_training()

def run_training():
    
    
    model = Net()
    print(model)
    model = model.cuda()
    # Print the number of trainable parameters
    pp=0
    for p in list(model.parameters()):
        nnd=1
        for s in list(p.size()):
            nnd = nnd*s
        pp += nnd
    print(pp)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    random_seed= 42
    shuffle_dataset = True
    dataset_size = 500
    batch_size = 64

    validation_split = .2
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    
    train_dataset  = audio_esc50(train=True, args=args, indices=train_indices)
    val_dataset = audio_esc50(train=True, args=args, indices=val_indices)
    train_indices = list(range(int(np.floor(dataset_size*(1-validation_split)))))
    val_indices =  list(range(int(np.floor(dataset_size*(validation_split)))))
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                    sampler=val_sampler)

    all_result = []
#     n_test = 3

#     for k in range(n_test):

    val_history = []
    val_loss_hist = []
    train_history = []
    train_loss_hist = []

    for epoch in range(600):  # loop over the dataset multiple times

    ########## Validation ###########
        model.eval()
        count = 0
        running_accuracy = 0
        running_loss = 0
        t1 = time.time()
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            count += len(labels)
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.cuda()
            labels = labels.cuda()
            if args.nCrops > 1:
#             if False:
                inputs = torch.reshape(inputs,(inputs.size(0)* args.nCrops, inputs.size(1)//args.nCrops))
                outputs = model(inputs)
                outputs = outputs.reshape((outputs.shape[0] // args.nCrops, args.nCrops, outputs.shape[1]))
                outputs = torch.mean(outputs, axis=1)
            else:
                outputs = model(inputs)
#             print(outputs.shape)
            _, preds = torch.max(outputs, 1)
#             print(preds.shape)
            preds = preds.float()
        
#             outputs = model(inputs)
#             val_loss = criterion(outputs, labels)
#             _, preds = torch.max(outputs, 1)
            acc_val = torch.eq(preds, labels).float().mean()
            running_accuracy += acc_val.item()*len(labels)
#             running_loss += val_loss.item()*len(labels)
            

        running_accuracy /= count
#         running_loss /= count
        val_history.append(running_accuracy)
#         val_loss_hist.append(running_loss)
        t2 = time.time()
        print("===========Phase: Val============")
        print("Validation Time: {}".format(t2 - t1))
        print("Epoch: {}  val_loss: {}".format(epoch, running_loss))
        print("Epoch: {}  val_accuracy: {}".format(epoch, running_accuracy))

    ######### Training ###########   
        
        model.train()
        running_loss = 0.0
        count = 0
        running_accuracy = 0
        t1 = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            count += len(labels)
            inputs = inputs.float()
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()*len(labels)
            _, preds = torch.max(outputs, 1)
            acc_train = torch.eq(preds, labels).float().mean()
            running_accuracy += acc_train.item()*len(labels)

        scheduler.step()
        running_accuracy /= count
        running_loss /= count
        train_history.append(running_accuracy)
        train_loss_hist.append(running_loss)
        t2 = time.time()
        print("===========Phase: Train============") 
        print("Training Time: {}".format(t2 - t1))
        print("Epoch: {}  train_loss: {}".format(epoch, running_loss))
        print("Epoch: {}  train_accuracy: {}".format(epoch, running_accuracy))
        print()

    print('Finished Training')

#     ## save models ##
#     if opt.save != 'None':
#         chainer.serializers.save_npz(
#             os.path.join(opt.save, 'model_split{}.npz'.format(split)), model)


if __name__ == '__main__':
    main()
