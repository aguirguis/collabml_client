import numpy as np
import torch

def get_mem_consumption(model, split_idx, freeze_idx, server_batch, client_batch):
    if freeze_idx < split_idx:          #we do not allow split after freeze index
        split_idx = freeze_idx
    a = torch.rand((1,3,224,224))
    input_size = np.prod(np.array(a.size()))*4/ (1024*1024)*server_batch
    x,begtosplit_sizes,_,_ = model(a,0,split_idx)
    intermediate_input_size = np.prod(np.array(x.size()))*4/ (1024*1024)*client_batch
    x,splittofreeze_sizes,_,_ = model(x,split_idx,freeze_idx)
    x,freezetoend_sizes,_,_ = model(x,freeze_idx,100)
    #Calculating the required sizes
    params=[param for param in model.parameters()]
    mod_sizes = [np.prod(np.array(p.size())) for p in params]
    model_size = np.sum(mod_sizes)*4/ (1024*1024)
    #note that before split, we do only forward pass so, we do not store gradients
    #after split index, we store gradients so we expect double the storage
    begtosplit_size = np.sum(begtosplit_sizes)/1024*server_batch
    splittofreeze_size = np.sum(splittofreeze_sizes)/1024*client_batch
    freezetoend_size = np.sum(freezetoend_sizes)/1024*client_batch

    total_server = input_size+model_size+begtosplit_size
    total_client = intermediate_input_size+model_size+splittofreeze_size+freezetoend_size*2
    vanilla = input_size*(client_batch/server_batch)+model_size+\
                (begtosplit_size*(client_batch/server_batch))+splittofreeze_size+freezetoend_size*2
    return total_server, total_client, vanilla
