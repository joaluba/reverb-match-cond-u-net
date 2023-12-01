import torch
from tqdm import tqdm
from datetime import datetime
import time
import sys
from torch.utils.tensorboard import SummaryWriter
import speechmetrics
import numpy as np
# my modules
import cond_waveunet_dataset
import cond_waveunet_loss
import cond_waveunet_model
from cond_waveunet_options import Options

def my_metrics_batch(taget_batch,prediction_batch,sr):
    metrics = speechmetrics.load(['stoi', 'pesq'], 3)
    for i in range(prediction_batch.shape[0]):
        scores = metrics(prediction_batch[0,0].detach().cpu().numpy(), taget_batch[0,0].detach().cpu().numpy(), rate=sr)
    return scores


def infer(model_reverbenc, model_waveunet, data, device):
    with torch.no_grad():
        # Function to infer target audio
        # ------------------------------
        # get datapoint
        sContent_in = data[0].to(device)
        sStyle_in=data[1].to(device)
        sTarget_gt=data[2].to(device)
        # forward pass - get prediction of the ir
        embedding_gt=model_reverbenc(sStyle_in)
        sTarget_prediction=model_waveunet(sContent_in,embedding_gt)
        return sContent_in, sStyle_in, sTarget_gt, sTarget_prediction
    

def eval_speechmetrics(model_reverbenc, model_waveunet,  testloader, args):

    device=args.device
    savedir = args.savedir
    store_outputs = args.store_outputs

    # save training parameters
    if (store_outputs):
        torch.save(args, savedir+'eval.pt')  

    # move components to device
    model_reverbenc=model_reverbenc.to(device)
    model_waveunet=model_waveunet.to(device)

    # allocate variable to track loss evolution 
    loss_evol=[]
    
    # ------------- TESTING START: -------------
    model_waveunet.eval() 
    model_reverbenc.eval()
    with torch.no_grad():
        score_ref_Content=0

        for j, data in tqdm(enumerate(testloader),total = len(testloader)):
            # infer and compute loss
            sContent_in, sStyle_in, sTarget_gt, sTarget_prediction=infer(model_reverbenc, model_waveunet, data, device)
            # compute metrics for each sample in the batch
            batchscore_ref_Content = my_metrics_batch(sTarget_gt, sContent_in, args.fs)
            batchscore_ref_Target = my_metrics_batch(sTarget_gt, sTarget_prediction, args.fs)

            sumscore_ref_Content+=np.sum(batchscore_ref_Content,axis=0)
            sumscore_ref_Target+=np.sum(batchscore_ref_Target,axis=0)

        avscore_ref_Content=sumscore_ref_Content/len(testloader.dataset)
        avscore_ref_Target=sumscore_ref_Target/len(testloader.dataset)
        
        return avscore_ref_Content, avscore_ref_Target




            
            


if __name__ == "__main__":
    # ---- test training loop ----

    args = Options().parse()

    # ---- MODEL: ----
    # load reverb encoder
    model_ReverbEncoder=cond_waveunet_model.ReverbEncoder(args)
    # laod waveunet 
    model_waveunet=cond_waveunet_model.waveunet(args)

    # ---- DATASET: ----
    args.split="test"
    testset=cond_waveunet_dataset.DatasetReverbTransfer(args)

    # create testloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)

    # move models to gpu
    model_ReverbEncoder.to(args.device)
    model_waveunet.to(args.device)

    # --------------------- Training: ---------------------
    metrics4content,metrics4target = eval_speechmetrics(model_ReverbEncoder, model_waveunet, testloader, args)

    print(metrics4content)








