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
    alfa_counter =0
    alfa_val = args.alfa
    beta_val = args.beta
    localIterCap = args.LocalIter

    #### APPROXIMATE THE NUMBER OF RUNS AND EPOCHS #####
    max_H = max(args.H_List)
    max_H = max([max_H,args.LocalIter])
    psuedo_run = math.ceil(N_s / (args.bs * num_client * max_H))
    total_train =  args.num_epoch * N_s / (args.bs * num_client * max_H)
    psuedo_epochs = math.ceil(total_train / psuedo_run)
    psuedo_lr_change = np.ceil(psuedo_epochs / args.num_epoch* np.asarray(args.lr_change))
    psuedo_alfa_change = np.ceil(psuedo_epochs / args.num_epoch* np.asarray(args.alfa_change))
    #####################################################
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[cl], milestones=psuedo_lr_change, gamma=0.1) for cl in
                  range(num_client)]
    print(psuedo_epochs,psuedo_lr_change,psuedo_alfa_change)
    for epoch in tqdm(range(psuedo_epochs)):
        if args.shuffle_dataset:
            sample_inds = dl.get_indices(trainset, args)

        if epoch in psuedo_alfa_change:
            ind = alfa_counter
            alfa_val = args.alfaList[ind]
            beta_val = args.betaList[ind]
            localIterCap = args.H_List[ind]
            alfa_counter += 1
            if args.synch: ###synch model
                [sf.pull_model(user,net_ps) for user in net_users]
        atWarmup = (args.warmUp and epoch < 5)
        if atWarmup:
            sf.lr_warm_up(optimizers,num_client,epoch,args.lr)
        runs = 1 if num_client == 1 else int(max_H * psuedo_run / localIterCap)
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
    return accuracys, losses

def train_global(args, device):###not finished yet!

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
    # optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in
    #               range(num_client)]

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
    gamma_val = args.gamma
    gamma_counter = 0
    localIterCap = args.LocalIter

    #### APPROXIMATE THE NUMBER OF RUNS AND EPOCHS #####
    max_H = max(args.H_List)
    max_H = max([max_H,args.LocalIter])
    psuedo_run = math.ceil(N_s / (args.bs * num_client * max_H))
    total_train =  args.num_epoch * N_s / (args.bs * num_client * max_H)
    psuedo_epochs = math.ceil(total_train / psuedo_run)
    psuedo_lr_change = np.ceil(psuedo_epochs / args.num_epoch* np.asarray(args.lr_change))
    psuedo_alfa_change = np.ceil(psuedo_epochs / args.num_epoch* np.asarray(args.alfa_change))
    #####################################################
    print(psuedo_epochs,psuedo_lr_change,psuedo_alfa_change)
    model_size = sf.count_parameters(net_ps)
    global_momentum = torch.zeros(model_size).to(device)
    currentLR = args.lr
    for epoch in tqdm(range(psuedo_epochs)):
        if args.shuffle_dataset: ## shuffle
            sample_inds = dl.get_indices(trainset, args)

        if epoch in args.lr_change: ## decay LR
            currentLR *= 0.1

        if epoch in psuedo_alfa_change:
            ind = gamma_counter
            gamma_val = args.gammaList[ind]
            localIterCap = args.H_List[ind]
            gamma_counter +=1
            if args.synch: ##synch models
                [sf.pull_model(user,net_ps) for user in net_users]


        atWarmup = (args.warmUp and epoch < 5)## check Warmup Conditions
        if atWarmup:##change LR
            if epoch == 0:
                currentLR = 0.1
            else:
                lr_change = (args.lr - 0.1) / 4
                currentLR = (lr_change * epoch) + 0.1


        runs = 1 if num_client == 1 else int(max_H * psuedo_run / localIterCap) ## calculate number of runs
        for run in range(runs):

            for cl in range(num_client):
                localIter = 0
                ## get local momentum value
                localM = global_momentum.mul(1 / localIterCap) if args.alg == 3 \
                    else global_momentum.mul(args.beta / localIterCap)

                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    flat_model = sf.get_model_flattened(net_users[cl],device)
                    if args.alg ==3:
                        flat_model.sub_(currentLR,localM)
                    elif args.alg ==4:
                        flat_model.sub_(currentLR * args.beta,localM)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    grad_flat = sf.get_grad_flattened(net_users[cl],device)
                    grad_flat.add_(1,args.wd) ## add weight_decay
                    flat_model.sub(currentLR,grad_flat)
                    if args.alg ==4:
                        localM.mul_(args.beta)
                        localM.add(grad_flat)
                    if args.alg != 1:
                        sf.make_model_unflattened(net_users[cl],flat_model,net_sizes,ind_pairs)
                    localIter +=1
                    if localIter == localIterCap and num_client > 1:
                        break
            ps_model_flat =sf.get_model_flattened(net_ps, device)
            avg_model_dif = torch.zeros_like(ps_model_flat).to(device)
            avg_grad = torch.zeros_like(ps_model_flat).to(device)
            avg_grad.add_(1,args.wd) ## add weight decay
            for cl in range(num_client):
                avg_grad.add_(1 / num_client, sf.get_grad_flattened(net_users[cl], device)) ## add grad values
                avg_model_dif.add_(1/num_client,(ps_model_flat.sub(1,sf.get_model_flattened(net_users[cl],device)))) ### send and get avg of dif values

            if args.alg !=1:## make workers into a circle
                global_momentum = avg_model_dif.mul(1 / currentLR)
                new_ps_model = ps_model_flat.sub(1 / currentLR, global_momentum)
                dif_norm = torch.norm(ps_model_flat.sub(1 , new_ps_model))
                sf.make_model_unflattened(net_ps, new_ps_model, net_sizes,
                                          ind_pairs)
                for worker_model in net_users:
                    worker_flat = sf.get_model_flattened(worker_model,device)
                    worker_norm = torch.norm(worker_flat.sub(1,new_ps_model))
                    constant = (dif_norm / worker_norm * gamma_val).item()
                    constant = np.min([constant,1])
                    worker_Dif = worker_flat.sub(1,new_ps_model)
                    sf.make_model_unflattened(worker_model, new_ps_model.add(constant,worker_Dif), net_sizes, ind_pairs)

            elif args.alg ==1: ## benchmark
                global_momentum.mul_(args.GlobalM)
                global_momentum.add_(1,avg_grad)
                sf.make_model_unflattened(net_ps, ps_model_flat.sub(currentLR,global_momentum), net_sizes, ind_pairs)
                [sf.pull_model(model,net_ps) for model in net_users]
            else:
                raise Exception('no such algorithm')

        acc = evaluate_accuracy(net_ps, testloader, device)
        loss = evaluate_training_loss(net_ps,trainLoader_all,device)
        accuracys.append(acc * 100)
        losses.append(loss)
        print('accuracy:{}, Loss:{}'.format(acc*100,loss))
        #[schedulers[cl].step() for cl in range(num_client)] ## adjust Learning rate
    return accuracys, losses



