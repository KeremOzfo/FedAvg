import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0, help='cuda:No')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='resnet18', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='iid', help='distribution of dataset; iid or non_iid')
    parser.add_argument('--numb_cls_usr', type=int, default=3, help='number of class per user if non_iid2 selected')

    # Federated params
    parser.add_argument('--mode', type=str, default='global', help='mode')
    parser.add_argument('--alg', type=int, default=3, help='alg type')
    parser.add_argument('--shuffle_dataset', type=bool, default=False, help='LR warm up.')
    parser.add_argument('--alfa', type=float, default=1, help='weigths ')
    parser.add_argument('--gamma', type=float, default=0, help='weigths ')
    parser.add_argument('--beta', type=float, default=0.9, help='weigths ')
    parser.add_argument('--alfa_change', type=tuple, default=[150, 160, 250],
                        help='determines the at which epochs alfa-beta-h changes')
    parser.add_argument('--alfaList', type=list, default=[1, 0.95, 0], help='weigths ')
    parser.add_argument('--gammaList', type=list, default=[0, 0.05, 1], help='weigths ')
    parser.add_argument('--H_List', type=list, default=[1, 8, 8], help='set to local Iter ')
    parser.add_argument('--num_client', type=int, default=16, help='number of clients')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--LocalIter', type=int, default=1, help='communication workers')
    parser.add_argument('--bs', type=int, default=128, help='batchsize')
    parser.add_argument('--lr', type=float, default=1.6, help='learning_rate')
    parser.add_argument('--nesterov', type=bool, default=True, help='enable nesterov momentum')
    parser.add_argument('--LocalM', type=float, default=0.9, help='momentum')
    parser.add_argument('--GlobalM', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay Value')
    parser.add_argument('--lr_change', type=list, default=[150, 250],
                        help='determines the at which epochs lr will decrease')
    parser.add_argument('--synch', type=bool, default=False, help='enable model synch at alfa change')
    parser.add_argument('--warmUp', type=bool, default=True, help='LR warm up.')

    args = parser.parse_args()
    return args
