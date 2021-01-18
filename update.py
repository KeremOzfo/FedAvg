def global_update_normalize(local_diff,max_diff_norm,args):
    # local_differences contains the local model diffrence of the workers.
    # max_diff_norm is the maximum norm value allowed for each local model diffrence.
    ############ Here we nomalize the local diffrences###################
    for i in range (args.numb_worker)
        global_diff+= (1/arg.numb_worker) local_diff[i] * (minimum(torch.norm(local_diff[i]),max_diff_norm)/torch.norm(local_diff[i]))
        max_diff_norm_new+= (1/arg.numb_worker)torch.norm(local_diff[i])
    max_diff_norm_new = max_diff_norm_new * args.norm_coeff + max_diff_norm (1-args.norm_coeff)
    return max_diff_norm_new, global_diff

def global_update_relaxed_average(local_differences, global_difference_prev,args):
    
    return  
