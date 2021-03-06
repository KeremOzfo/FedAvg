from train_funcs import *
import numpy as np
from parameters import *
import torch
import random
import datetime
import os

device = torch.device("cpu")
args = args_parser()

if __name__ == '__main__':
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    simulation_ID = int(random.uniform(1,999))
    print('device:',device)
    args = args_parser()
    for arg in vars(args):
       print(arg, ':', getattr(args, arg))
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    simulation = 'mode_{}-ALG_{}-shuffle_{}-H_{}{}'.format(args.mode,args.alg,args.shuffle_dataset,args.LocalIter,args.H_List)
    newFile = date + simulation + '-sim_ID-' + str(simulation_ID)
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results', newFile)
    n_path_acc = n_path + '/acc'
    n_path_loss = n_path + '/loss'
    for i in range(5):
        if args.mode == 'global':
            accs, loss = train_global(args, device)
        else:
            accs, loss = train(args, device)
        if i == 0:
            os.mkdir(n_path)
            os.mkdir(n_path_acc)
            os.mkdir(n_path_loss)
            f = open(n_path + '/simulation_Details.txt', 'w+')
            f.write('simID = ' + str(simulation_ID) + '\n')
            f.write('############## Args ###############' + '\n')
            for arg in vars(args):
                line = str(arg) + ' : ' + str(getattr(args, arg))
                f.write(line + '\n')
            f.write('############ Results ###############' + '\n')
            f.close()
        s_loc = date + f'federated_avg_acc' + '--' + str(i)
        s_loc2 = date + f'federated_avg_loss' + '--' + str(i)
        s_loc = os.path.join(n_path_acc,s_loc)
        s_loc2 = os.path.join(n_path_loss, s_loc2)
        np.save(s_loc,accs)
        np.save(s_loc2, loss)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f.write('Trial ' + str(i) + ' results at ' + str(accs[len(accs)-1]) + '\n')
        f.close()