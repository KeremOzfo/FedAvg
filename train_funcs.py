import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
import server_functions as sf
import math
from parameters import *
import time
import numpy as np
from tqdm import tqdm


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def evaluate_training_loss(model, trainloader,device):
    criterion = nn.CrossEntropyLoss()
    count = 0
    loss = 0
    with torch.no_grad():
        for data in trainloader:
            count+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
    return loss/count


def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]
    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4,momentum=args.LocalM,nesterov=args.nesterov) for cl in
                  range(num_client)]
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[cl], milestones=args.lr_change, gamma=0.1) for cl in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)
    trainLoader_all = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []
    losses =[]
    acc = evaluate_accuracy(net_ps, testloader, device)
    loss = evaluate_training_loss(net_ps, trainLoader_all, device)
    accuracys.append(acc * 100)
    losses.append(loss)
    assert N_s/num_client > args.LocalIter * args.bs
    alfa_val = args.alfa
    beta_val = args.beta
    localIterCap = args.LocalIter
    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = (args.warmUp and epoch < 5)
        if atWarmup:
            sf.lr_warm_up(optimizers,num_client,epoch,args.lr)

        runs = math.ceil(N_s / (args.bs * num_client * localIterCap))
        for run in range(runs):
            for cl in range(num_client):
                localIter = 0

                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizers[cl].zero_grad()
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    optimizers[cl].step()
                    localIter +=1
                    if localIter == localIterCap and num_client > 1:
                        break
            if atWarmup:
                alfa_val = 1
                beta_val = 0
            ps_model_flat =sf.get_model_flattened(net_ps, device)
            avg_model = torch.zeros_like(ps_model_flat)
            for cl in range(num_client):
                avg_model.add_(1/num_client, sf.get_model_flattened(net_users[cl],device))
            sf.make_model_unflattened(net_ps, avg_model, net_sizes, ind_pairs)
            if args.alg == 1:
                for cl in range(num_client):
                    worker_model = torch.zeros_like(ps_model_flat)
                    worker_model.add_(1-alfa_val,sf.get_model_flattened(net_users[cl],device))
                    worker_model.add_(alfa_val, avg_model)
                    sf.make_model_unflattened(net_users[cl], worker_model, net_sizes, ind_pairs)
            elif args.alg == 2:
                dif_norm = torch.norm(avg_model.sub(1,ps_model_flat))
                for worker_model in net_users:
                    worker_flat = sf.get_model_flattened(worker_model,device)
                    worker_norm = torch.norm(worker_flat.sub(1,avg_model))
                    constant = (dif_norm / worker_norm * beta_val).item()
                    constant = np.min([constant,1])
                    worker_Dif = worker_flat.sub(1,avg_model)
                    sf.make_model_unflattened(worker_model, avg_model.add(constant,worker_Dif), net_sizes, ind_pairs)
            elif args.alg ==3: ## benchmark
                avg_dif = torch.zeros_like(ps_model_flat)
                [avg_dif.add_(1/num_client,(sf.get_model_flattened(model,device)).sub(1,ps_model_flat)) for model in net_users]
                sf.make_model_unflattened(net_ps, ps_model_flat.add(1,avg_dif), net_sizes, ind_pairs)
                [sf.pull_model(model,net_ps) for model in net_users]
            else:
                raise Exception('no such algorithm')

        acc = evaluate_accuracy(net_ps, testloader, device)
        loss = evaluate_training_loss(net_ps,trainLoader_all,device)
        accuracys.append(acc * 100)
        losses.append(loss)
        print('accuracy:{}, Loss:{}'.format(acc*100,loss))
        [schedulers[cl].step() for cl in range(num_client)] ## adjust Learning rate
        if epoch in args.alfa_change:
            lrs = np.asarray(args.alfa_change)
            ind = np.where(lrs == epoch)[0][0]
            alfa_val = args.alfaList[ind]
            beta_val = args.betaList[ind]
            localIterCap = args.H_List[ind]
            if args.synch:
                [sf.pull_model(user,net_ps) for user in net_users]
    return accuracys, losses



