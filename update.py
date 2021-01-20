#Model update method with normalization
def global_update_normalized(local_diff,max_diff_norm,args):
    # local_differences contains the local model diffrence of the workers.
    # max_diff_norm is the maximum norm value allowed for each local model diffrence.
    ############ Here we nomalize the local diffrences###################
    for i in range (args.numb_worker):
    ####################### Global diffrence is the average of normalized local diffrences ##########################################
        global_diff+= (1/arg.numb_worker) local_diff[i] * (minimum(torch.norm(local_diff[i]),max_diff_norm)/torch.norm(local_diff[i]))
        max_diff_norm_new+= (1/arg.numb_worker)torch.norm(local_diff[i])
    max_diff_norm_new = max_diff_norm_new * args.norm_coeff + max_diff_norm (1-args.norm_coeff)
    return max_diff_norm_new , global_diff
# Model update method with relaxed averaging
def global_update_relaxed_average(local_diff, global_diff_prev,global_model,args):
     for i in range (args.num_client):
    ####################### Global diffrence is the average of  local diffrences ##########################################
        global_diff+= (1/arg.numb_worker) local_diff[i]
     global_model = global_model + global_diff
    return  global_model
# Local Model update with relaxed averaging
def local_update_relaxed_average():
    #Burda local model update edilcek###########
