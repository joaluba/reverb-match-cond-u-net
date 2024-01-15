import torch
from tqdm import tqdm
from datetime import datetime
import time
import os
from torch.utils.tensorboard import SummaryWriter
# my modules
import cond_waveunet_dataset
import cond_waveunet_loss
import cond_waveunet_model
from cond_waveunet_options import Options


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
    
    
def infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, reverb_criterion, data,device):
    # Function to infer target and compute loss - same for training, testing and validation
    # -------------------------------------------------------------------------------------
    # get datapoint
    sContent_in = data[0].to(device)
    sStyle_in=data[1].to(device)
    sTarget_gt=data[2].to(device)
    sAnecho=data[3].to(device)
    # forward pass - get prediction of the ir
    embedding_gt=model_reverbenc(sStyle_in)
    sTarget_prediction=model_waveunet(sContent_in,embedding_gt)
    # compute embedding of the predicted waveform 
    embedding_prediction=model_reverbenc(sTarget_prediction)
    # loss based on full audio signal
    L_sc, L_mag = audio_criterion(sTarget_gt.squeeze(1), sTarget_prediction.squeeze(1))
    # loss based on reverb tail
    L_sc_rev, L_mag_rev = reverb_criterion(sTarget_gt.squeeze(1)-sAnecho.squeeze(1), sTarget_prediction.squeeze(1)-sAnecho.squeeze(1))
    # loss based on reverb tail
    L_emb=(1-((torch.mean(emb_criterion(embedding_gt,embedding_prediction))+ 1) / 2))

    # loss 
    loss=L_sc + L_mag + L_sc_rev + L_mag_rev + L_emb
    return loss


def train_and_test(model_reverbenc, model_waveunet, trainloader, valloader, testloader, args):

    # training parameters
    device = args.device
    criterion=cond_waveunet_loss.LossOfChoice(args)

    if args.optimizer=="adam":
        optimizer_waveunet =  torch.optim.AdamW(model_waveunet.parameters(), args.learn_rate)
        optimizer_reverbenc =  torch.optim.AdamW(model_reverbenc.parameters(), args.learn_rate)

    num_epochs = args.num_epochs
    savedir = args.savedir
    store_outputs = args.store_outputs

    # initialize tensorboard writer 
    writer=SummaryWriter(savedir) 
    # save training parameters
    if (store_outputs):
        torch.save(args, os.path.join(savedir,'trainargs.pt'))


    # allocate variable to track loss evolution 
    loss_evol=[]
    best_val_loss=float("inf")
    
    # ------------- TRAINING START: -------------
    start = time.time()
    for epoch in range(num_epochs):

        # ----- Training loop for this epoch: -----
        model_waveunet.train()
        model_reverbenc.train()
        train_loss=0
        end = time.time()
        for j,data in tqdm(enumerate(trainloader),total = len(trainloader)):

            # # measure data loading time (how much time of the training loop is spent on waiting on the next batch)
            # # - should be zero if the data loading is not a bottleneck
            # print(f"Time to load data: {time.time() - end}")     

            # infer and compute loss
            loss=criterion(data, model_waveunet, model_reverbenc, device)
            # empty gradient
            optimizer_waveunet.zero_grad()
            optimizer_reverbenc.zero_grad()
            # compute gradients 
            loss.backward()
            # update weights
            optimizer_waveunet.step()
            optimizer_reverbenc.step()
            # compute loss for the current batch
            train_loss += loss.item()
            writer.add_scalar('TrainLossPerBatch', loss.item()/args.batch_size, epoch * len(trainloader) + j) # tensorboard

            # lod audios to tensorboard:
            if j==0:
                _,_,sTarget_gt, sTarget_prediction=infer(model_reverbenc, model_waveunet, data, args.device)
                for i in range(5):
                    wave_target=sTarget_gt[i,:,:].squeeze(0)
                    wave_predict=sTarget_prediction[i,:,:].squeeze(0)
                    writer.add_audio(f'Target_dp{i}', wave_target/wave_target.abs().max(), torch.tensor(args.fs))
                    writer.add_audio(f'Predict_dp{i}', wave_predict/wave_predict.abs().max(), torch.tensor(args.fs))

            # # measure time required to process one batch 
            # print(f"Time to process a batch: {time.time() - end}")
            # end = time.time()  

        # ----- Validation loop for this epoch: -----
        model_waveunet.eval() 
        model_reverbenc.eval()
        with torch.no_grad():
            val_loss=0
            for j,data in tqdm(enumerate(valloader),total = len(valloader)):
                            
                # infer and compute loss
                loss=criterion(data, model_waveunet, model_reverbenc, device)
                # compute loss for the current batch
                val_loss += loss.item()

               
        # Print stats at the end of the epoch
        num_samples_train=len(trainloader.sampler)
        num_samples_val=len(valloader.sampler)
        avg_train_loss = train_loss / num_samples_train
        avg_val_loss = val_loss / num_samples_val
        loss_evol.append((avg_train_loss,avg_val_loss)) # tensorboard
        writer.add_scalar('TrainLoss', avg_train_loss, epoch) # tensorboard
        writer.add_scalar('ValLoss', avg_val_loss, epoch)
        print(f'Epoch: {epoch}, Train. Loss: {avg_train_loss:.5f}, Val. Loss: {avg_val_loss:.5f}')
        
        # Save checkpoint (overwrite)
        if (store_outputs) & (epoch % args.checkpoint_step ==0):
            torch.save({
                        'epoch': epoch,
                        'model_waveunet_state_dict': model_waveunet.state_dict(),
                        'model_reverbenc_state_dict': model_reverbenc.state_dict(),
                        'optimizer_waveunet_state_dict': optimizer_waveunet.state_dict(),
                        'optimizer_reverbenc_state_dict': optimizer_reverbenc.state_dict(),
                        'loss': loss_evol,
                        }, os.path.join(savedir,'checkpoint' +str(epoch)+'.pt'))
            
        # Early stopping: stop when validation loss doesnt improve for 10 epochs
        if avg_val_loss < best_val_loss:
            # save the best condition so far 
            if (store_outputs):
                torch.save({
                    'epoch': epoch,
                    'model_waveunet_state_dict': model_waveunet.state_dict(),
                    'model_reverbenc_state_dict': model_reverbenc.state_dict(),
                    'optimizer_waveunet_state_dict': optimizer_waveunet.state_dict(),
                    'optimizer_reverbenc_state_dict': optimizer_reverbenc.state_dict(),
                    'loss': loss_evol,
                    }, os.path.join(savedir,'checkpoint_best.pt'))

            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            print(f'Loss did not decrease x{counter}.')

        if counter >= 15:
            print(f'Early stopping after {counter +1} epochs without improvement.')
            break
        

            
  
    end=time.time()
    print(f"Finished training after: {(end-start)} seconds")

    writer.close() # tensorboard  

    # ------------- TESTING START: -------------
    model_waveunet.eval() 
    model_reverbenc.eval()
    with torch.no_grad():
        test_loss=0
        for j,data in tqdm(enumerate(testloader),total = len(testloader)):
            # infer and compute loss
            loss=criterion(data, model_waveunet, model_reverbenc, device)
            # compute loss for the current batch
            test_loss += loss.item()


    
    # Print stats at the end of the epoch
    num_samples_test=len(testloader.sampler)
    avg_test_loss = test_loss / num_samples_test
    print(f'Test. Loss: {avg_test_loss:.5f}')
    
    # log audios to tensorboard



if __name__ == "__main__":
    # ---- test training loop ----

    args = Options().parse()

    # ---- MODEL: ----
    # load reverb encoder
    model_ReverbEncoder=cond_waveunet_model.ReverbEncoder(args)
    # laod waveunet 
    model_waveunet=cond_waveunet_model.waveunet(args)

    # ---- DATASET: ----
    args.split="train"
    trainset=cond_waveunet_dataset.DatasetReverbTransfer(args)
    args.split="test"
    testset=cond_waveunet_dataset.DatasetReverbTransfer(args)
    args.split="val"
    valset=cond_waveunet_dataset.DatasetReverbTransfer(args)

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)

    # move models to gpu
    model_ReverbEncoder.to(args.device)
    model_waveunet.to(args.device)

    # --------------------- Training: ---------------------
    train_and_test(model_ReverbEncoder, model_waveunet, trainloader, valloader, testloader, args)








