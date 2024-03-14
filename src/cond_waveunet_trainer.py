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

class Trainer(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.load_train_objects(args)
        self.args=args

    def load_train_objects(self,args):
        # ---- MODELS: ----
        # load reverb encoder
        self.model_reverbenc=cond_waveunet_model.ReverbEncoder(args).to(args.device)
        # laod waveunet 
        self.model_waveunet=cond_waveunet_model.waveunet(args).to(args.device)

        # ---- OPTIMIZERS: ----
        if args.optimizer=="adam":
            self.optimizer_waveunet =  torch.optim.AdamW(self.model_waveunet.parameters(), args.learn_rate)
            self.optimizer_reverbenc =  torch.optim.AdamW(self.model_reverbenc.parameters(), args.learn_rate)

        # ---- LOSS CRITERION: ----
        self.criterion=cond_waveunet_loss.LossOfChoice(args).to(args.device)

        # ---- DATASETS: ----
        args.split="train"
        self.trainset=cond_waveunet_dataset.DatasetReverbTransfer(args)
        args.split="val"
        self.valset=cond_waveunet_dataset.DatasetReverbTransfer(args)

        # ---- DATA LOADERS: ----
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)

        # ---- PREPARE TRAINING LOOP (starting from scratch or resume training): ----
        if args.resume_checkpoint==None:
            print("Starting new training from scratch in : "+ args.savedir)
            self.writer=SummaryWriter(args.savedir) 
            self.loss_evol=[]
            self.start_epoch=0
            self.best_val_loss=float("inf")
        else:
            checkpoint_path=os.path.join(args.savedir,args.resume_checkpoint)
            train_results=torch.load(checkpoint_path,map_location=args.device)
            self.model_reverbenc.load_state_dict(train_results["model_reverbenc_state_dict"])           
            self.model_waveunet.load_state_dict(train_results["model_waveunet_state_dict"])
            self.optimizer_reverbenc.load_state_dict(train_results["optimizer_reverbenc_state_dict"])
            self.optimizer_waveunet.load_state_dict(train_results["optimizer_waveunet_state_dict"])
            self.writer=SummaryWriter(args.resume_tboard)
            self.loss_evol = train_results['loss']
            self.start_epoch = train_results['epoch']
            self.best_val_loss=float("inf")
            print("Resume training from epoch "+ str(self.start_epoch) + " from training "+ args.savedir)


    def infer(self,data):
        with torch.no_grad():
            # Function to infer target audio
            # ------------------------------
            # get datapoint
            sContent_in = data[0].to(self.args.device)
            sStyle_in=data[1].to(self.args.device)
            sTarget_gt=data[2].to(self.args.device)
            # forward pass - get prediction of the ir
            embedding_gt=self.model_reverbenc(sStyle_in)
            sTarget_prediction=self.model_waveunet(sContent_in,embedding_gt)
            return sContent_in, sStyle_in, sTarget_gt, sTarget_prediction
        
    def logaudio_tboard(self,writer):
        with torch.no_grad():
            chosen_idx=[9372,49836,7247,9950,473]
            for i in range(0,len(chosen_idx)):
                data=self.trainset[chosen_idx[i]]
                data = [data[i].unsqueeze(0) for i in range(len(data))]
                _,_,sTarget_gt, sTarget_prediction=self.infer(data)
                wave_target=sTarget_gt[0,:,:].squeeze(0)
                wave_predict=sTarget_prediction[0,:,:].squeeze(0)
                writer.add_audio(f'Target_dp{i}', wave_target/wave_target.abs().max(), torch.tensor(self.args.fs))
                writer.add_audio(f'Predict_dp{i}', wave_predict/wave_predict.abs().max(), torch.tensor(self.args.fs))

    def save_checkpoint(self,epoch,loss_evol,name):
        torch.save({
                    'epoch': epoch,
                    'model_waveunet_state_dict': self.model_waveunet.state_dict(),
                    'model_reverbenc_state_dict': self.model_reverbenc.state_dict(),
                    'optimizer_waveunet_state_dict': self.optimizer_waveunet.state_dict(),
                    'optimizer_reverbenc_state_dict': self.optimizer_reverbenc.state_dict(),
                    'loss': loss_evol,
                    }, os.path.join(self.args.savedir,'checkpoint' +name+'.pt'))
        
    
    def train(self):

        if (bool(self.args.store_outputs)):
            torch.save(self.args, os.path.join(self.args.savedir,'trainargs.pt'))
        
        # ------------- TRAINING START: -------------
        start = time.time()
        for epoch in range(self.start_epoch,self.args.num_epochs):

            # ----- Training loop for this epoch: -----
            self.model_waveunet.train()
            self.model_reverbenc.train()
            train_loss=0
            end = time.time()
            for j,data in tqdm(enumerate(self.trainloader),total = len(self.trainloader)):

                # # measure data loading time (how much time of the training loop is spent on waiting on the next batch)
                # # - should be zero if the data loading is not a bottleneck
                # print(f"Time to load data: {time.time() - end}")     

                # infer and compute loss
                loss_vals,loss_names=self.criterion(data, self.model_waveunet, self.model_reverbenc, self.args.device)
                # log and sum all loss terms
                loss=0
                for i in range(0,len(loss_vals)):
                    loss_term=self.args.loss_alphas[i]*loss_vals[i]
                    loss+=loss_term
                    self.writer.add_scalar(loss_names[i], loss_term.item(), epoch * len(self.trainloader) + j) # tensorboard
                    

                # empty gradient
                self.optimizer_waveunet.zero_grad()
                self.optimizer_reverbenc.zero_grad()
                # compute gradients 
                loss.backward()
                # update weights
                self.optimizer_waveunet.step()
                self.optimizer_reverbenc.step()
                # compute loss for the current batch
                train_loss += loss.item()

                # log variables and audios to tensorboard:
                self.writer.add_scalar('TrainLossPerBatch', loss.item(), epoch * len(self.trainloader) + j) # tensorboard

                if j==0:
                    self.logaudio_tboard(self.writer)# tensorboard

                # # measure time required to process one batch 
                # print(f"Time to process a batch: {time.time() - end}")
                # end = time.time()  

            # ----- Validation loop for this epoch: -----
            self.model_waveunet.eval() 
            self.model_reverbenc.eval()
            with torch.no_grad():
                val_loss=0
                for j,data in tqdm(enumerate(self.valloader),total = len(self.valloader)):
                                
                    # infer and compute loss
                    loss_vals,loss_names==self.criterion(data, self.model_waveunet, self.model_reverbenc, self.args.device)   
                    # log and sum all loss terms
                    loss=0
                    for i in range(0,len(loss_vals)):
                        loss_term=self.args.loss_alphas[i]*loss_vals[i]
                        loss+=loss_term
                        self.writer.add_scalar("val_"+loss_names[i], loss_term.item(), epoch * len(self.valloader) + j) # tensorboard

                    # compute loss for the current batch
                    val_loss += loss.item()  

            # Print stats at the end of the epoch
            num_samples_train=len(self.trainloader.sampler)
            num_samples_val=len(self.valloader.sampler)
            avg_train_loss = train_loss / num_samples_train
            avg_val_loss = val_loss / num_samples_val
            print(f'Epoch: {epoch}, Train. Loss: {avg_train_loss:.5f}, Val. Loss: {avg_val_loss:.5f}')
            self.loss_evol.append((avg_train_loss,avg_val_loss)) 

            self.writer.add_scalar('TrainLoss', avg_train_loss, epoch) # tensorboard
            self.writer.add_scalar('ValLoss', avg_val_loss, epoch) # tensorboard
            
            # Save checkpoint (overwrite)
            if (bool(self.args.store_outputs)) & (epoch % self.args.checkpoint_step ==0):
                self.save_checkpoint(epoch,self.loss_evol,str(epoch))
                
            # Early stopping: stop when validation loss doesnt improve for 30 epochs
            if avg_val_loss < self.best_val_loss:
                # save the best checkpoint so far 
                if (bool(self.args.store_outputs)):
                    self.save_checkpoint(epoch,self.loss_evol,"best")
                self.best_val_loss = avg_val_loss
                counter = 0
                print(f'Loss decreased.')
            else:
                counter += 1
                print(f'Loss did not decrease for {counter} epochs.')
            
        end=time.time()
        print(f"Finished training after: {(end-start)} seconds")
        self.writer.close()

if __name__ == "__main__":
    # ---- test training loop ----

    args = Options().parse()
    new_experiment=Trainer(args)
    new_experiment.train()














