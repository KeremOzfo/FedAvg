#Model update method with normalization
import torch
def global_update_normalized(local_diffs,max_diff_norm,device,args):
    # local_differences contains the local model diffrence of the workers.
    # max_diff_norm is the maximum norm value allowed for each local model diffrence.
    #local diff set of difs
    ############ Here we nomalize the local diffrences###################
    global_diff = torch.zeros_like(local_diffs[0], device=device)
    max_diff_norm_new = torch.zeros_like(local_diffs[0], device=device)
    for dif in local_diffs:
    ####################### Global diffrence is the average of normalized local diffrences ##########################################
        global_diff = global_diff.add(dif* min([torch.norm(dif),max_diff_norm]) / torch.norm(dif),alpha=1/args.num_client)
        max_diff_norm_new = max_diff_norm_new.add(torch.norm(dif),alpha=1/args.num_client)
    max_diff_norm_new = max_diff_norm_new * args.norm_coeff + max_diff_norm (1-args.norm_coeff)
    return max_diff_norm_new , global_diff
# Model update method with relaxed averaging
def global_update_relaxed_average(local_diffs,global_model,device,args):
     global_diff = torch.zeros_like(global_model,device=device)
     for dif in local_diffs:
        global_diff =global_diff.add(dif , alpha=1/args.num_client)
     global_model = global_model.add(global_diff, alpha=1)
     return global_model
# Local Model update with relaxed averaging
def local_update_relaxed_average():
    #Burda local model update edilcek###########
