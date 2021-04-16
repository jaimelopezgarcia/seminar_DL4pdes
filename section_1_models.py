import os
import torch
from torch import nn
import skimage
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
#from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from tqdm import tqdm
import numpy as np
from utils import make_simulation_gif
from utils import plot_phases
from utils import eval_sim
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels , n_layers = 2, stride = 1, padding = 1, normalization = True):
        
        super().__init__()
        
        layers = []
        
        for i in range(n_layers):
            
            if i == 0:
                _in_channels = in_channels
                _stride = stride
            else:
                _in_channels = out_channels
                _stride = 1
                
            layer = nn.Conv2d(_in_channels, out_channels, kernel_size=3, stride = _stride,
                     padding=padding, bias=True) #Bias can be set to false if using batch_norm ( is present there)
            
            torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            
            if normalization:
                _layer = torch.nn.BatchNorm2d(out_channels)
                
                layer = nn.Sequential(*[layer, _layer]) 
            
            layers.append(layer)
            
            
            
        self._layers = nn.ModuleList(layers)
        
        if (in_channels != out_channels) or (stride>1):
            
            self._shortcut = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = padding)
        
        else:
            
            self._shortcut = nn.Identity()
            
        self._activation = torch.nn.ReLU()
        
    def forward(self, x):
        
        _x = x
        
        for layer in self._layers:
            
            _x = self._activation(layer(_x))
           
        out = self._shortcut(x) + _x
        
        return out
    
    
class BasicNet(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, blocks = [2, 2, 2, 2, 2], add_input_output = True, normalization = True):
        
        super().__init__()
        
        layers = []
        
        for i,_block in enumerate(blocks):
            
            
            if i == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels
                
                
            layer = ResidualBlock(_in_channels, hidden_channels, stride = 1, padding=1, normalization = normalization)
            
            layers.append(layer)
            

            
        self._hidden_layers = nn.ModuleList(layers)
        
        
        self._out_layer = nn.Conv2d( hidden_channels , out_channels, kernel_size=1, stride = 1,
             padding=0, bias=True)
        
        self._add_input_output = add_input_output
        
        
        self.act_out = torch.nn.Tanh()
        
        
    def forward(self, x):
        
        _x = x
        
        for layer in self._hidden_layers:
            
            _x = layer(_x)
            
        if self._add_input_output:
            
            _x = self._out_layer(_x) + x
           
        else:
            
            _x = self._out_layer(_x)
         
        #_x = self.act_out(_x)
        return _x
    
    
    
class BasicNetSkipCon(nn.Module):
    
    """
    Symmetrical skip connections between blocks
    
    
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, blocks = [2, 2, 2, 2, 2], normalization = True):
        
        super().__init__()
        
        layers = []
        
        for i,_block in enumerate(blocks):
            
            
            if i == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels
                
                
            layer = ResidualBlock(_in_channels, hidden_channels, stride = 1, padding=1, normalization = normalization)
            
            layers.append(layer)
            

            
        self._hidden_layers = nn.ModuleList(layers)
        
        
        self._out_layer = nn.Conv2d( hidden_channels , out_channels, kernel_size=3, stride = 1,
             padding=1, bias=True)
        
        torch.nn.init.kaiming_uniform_(self._out_layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                
        #symmetrical skip connections, make pairs
        
        
        layer_num = np.arange(0,len(self._hidden_layers),1)
        pairs = {}
        
        for i,ln in enumerate(self._hidden_layers):
            
            j = len(self._hidden_layers)-i-1
    
            if j <= i:
                break
            
            pairs[i] = j
        
        self._pairs = {val:key for key,val in pairs.items() }
        
        
    def forward(self, x):
        
        _x = x
        
        outs = []
        
        for i,layer in enumerate(self._hidden_layers):
            
            _x = layer(_x)
            
            outs.append(_x)
            
            if i in self._pairs:
                _x = _x + outs[self._pairs[i]]
        

        _x = self._out_layer(_x) + x

        return _x  
    
class ModelSim(pl.LightningModule):
    
    def __init__(self, model, *args_model, results_dir = ".", tol_next_step = 0.015 , lr = 1e-3,  **kwargs_model):
        super().__init__()
        
        self._model = model(*args_model, **kwargs_model)
        self.criterion = torch.nn.L1Loss()
        self._results_dir = results_dir
        self._tol_next_step = tol_next_step #if error less than this adds next step
        self._n_steps_ahead = 0
        self._lr = lr
        
        try:
            os.makedirs(self._results_dir)
        except:
            pass
        
    def forward(self,x):
        
        x = self._model(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        

        
        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"],batch["Y"][:,0,:,:,:],batch["Y"][:,1,:,:,:],batch["Y"][:,2,:,:,:],batch["Y"][:,3,:,:,:]
                
        Ydata = [Ystep1, Ystep2, Ystep3, Ystep4]
        
        Ypred1 = self(Xb)
        
        loss1 = self.criterion(Ypred1, Ystep1)
        
        Ypred = Ypred1
        
        losses = []
        losses.append(loss1)
        
        for i in range(1,self._n_steps_ahead):
            
            Ypred = self(Ypred)
            
            losses.append(self.criterion(Ypred, Ydata[i]))
        
        losses = [l.view(1) for l in losses]
        loss = torch.mean( torch.cat(losses, 0) )

        return {"loss":loss, "log":{"train_loss": loss, "loss_s1": losses[0], "n_steps_ahead": self._n_steps_ahead}}
    
    def training_epoch_end(self, train_step_results):
        
        epoch_training_loss = torch.mean(torch.Tensor([d["loss"] for d in train_step_results]))
        

            
        return {"log": {"epoch_training_loss": epoch_training_loss } }
    
    def validation_epoch_end(self, validation_step_outputs):
        
        validation_step_outputs = np.array(torch.mean(torch.Tensor(validation_step_outputs),axis =0))
        
        val_loss1 = validation_step_outputs[0]
        val_loss2 = validation_step_outputs[1]
        val_loss3 = validation_step_outputs[2]
        val_loss4 = validation_step_outputs[3]
        
        
        if (val_loss1 < self._tol_next_step) and self._n_steps_ahead <=3:
            
            self._n_steps_ahead +=1
        
        return {
                "val_loss":val_loss1, 
            
                "progress_bar":{"val_loss_s1": val_loss1},
                
                "log":{"val_loss_1_log":val_loss1,
                      "val_loss_2_log":val_loss2,
                      "val_loss_3_log":val_loss3,
                      "val_loss_4_log":val_loss4,}
               }
    
        
        
    def validation_step(self, batch, batch_idx):
        
        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"],batch["Y"][:,0,:,:,:],batch["Y"][:,1,:,:,:],batch["Y"][:,2,:,:,:],batch["Y"][:,2,:,:,:]
        
        Ypred1 = self(Xb)
        Ypred2 = self(Ypred1)
        Ypred3 = self(Ypred2)
        Ypred4 = self(Ypred3)
        
        val_loss1 = self.criterion(Ypred1, Ystep1)
        val_loss2 = self.criterion(Ypred2, Ystep2)
        val_loss3 = self.criterion(Ypred3, Ystep3)
        val_loss4 = self.criterion(Ypred4, Ystep4)
        

        
        return [val_loss1,val_loss2,val_loss3,val_loss4]
    
    
    def test_step(self, batch, batch_idx):
        
        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"],batch["Y"][:,0,:,:,:],batch["Y"][:,1,:,:,:],batch["Y"][:,2,:,:,:],batch["Y"][:,2,:,:,:]
        
        Ypred1 = self(Xb)
        Ypred2 = self(Ypred1)
        Ypred3 = self(Ypred2)
        Ypred4 = self(Ypred3)
        
        val_loss1 = self.criterion(Ypred1, Ystep1)
        val_loss2 = self.criterion(Ypred2, Ystep2)
        val_loss3 = self.criterion(Ypred3, Ystep3)
        val_loss4 = self.criterion(Ypred4, Ystep4)
        

        
        return {"loss_s1": val_loss1, "loss_s2": val_loss2, "loss_s3": val_loss3, "loss_s4":val_loss4}
        

        
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self._lr)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold = 1e-3 ,verbose = True, eps = 1e-6)
        

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'val_loss'
        }
        
        return [optimizer], [scheduler]
    
    
    
    
    
    
    
    
    
####################### DATA UTILS####################
  
    
    
    
    
def load_data(data_dir, resize = (60,60), max_data = None):
    
    data_files = [name for name in os.listdir(data_dir) if "array" in name]
    
    if max_data:
        
        data_files = data_files[:max_data]
        
        
    data_arrays = [np.load(os.path.join(data_dir,file)) for file in tqdm(data_files)]
    
                                        
    if resize:
         
        for i,array in tqdm(enumerate(data_arrays)):
            
            data_arrays[i] = np.reshape(skimage.transform.resize(array, (len(array), resize[0], resize[1] )), (len(array), 1, resize[0], resize[1]) )
        
    names = data_files
    
    
    return names, data_arrays


def prepare_x_y( simulations , skip_steps = 10, store_steps_ahead = 5):
    
    X = []
    Y = []
    
    for simulation in tqdm(simulations):
        
        sim = simulation[2:] #remove first spiky snapshots

        lsim = len(sim) 
        
        for i in range(int( np.floor( lsim/skip_steps) )-store_steps_ahead ):
        
            s = i*skip_steps
            
            _Y = []
            

            for j in range(1,store_steps_ahead):
                sj = (i+j)*skip_steps
                _Y.append(sim[sj])
                
            _Y = np.array(_Y)
            
            
            X.append(sim[s])
            Y.append(_Y)
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X,Y

class SimDataset(Dataset):
    def __init__(self, X,Y):

        self._X = X
        self._Y = Y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):

        sample = {"X": self._X[idx], "Y": self._Y[idx]}
        
        return sample
    
    
class DataModule():#problemdatamoduele import
    
    def __init__(self, data_dir ,resize = (60,60), max_data = None, n_test_simulation = 12):
        
        super().__init__()
        
        self._max_data = max_data
        self._data_dir = data_dir
        self._names, self._data_arrays = None, None
        self._resize = resize
        self._n_test_simulation = n_test_simulation
        
    def prepare_data(self):
        
        self._names, self._data_arrays = load_data(self._data_dir, resize = self._resize,
                                                  max_data = self._max_data)
        
    def setup(self, skip_steps = 10, store_steps_ahead = 5, test_ratio = 0.2):
        
        self._skip_steps = skip_steps
        self._store_steps_ahead = store_steps_ahead
        
        n_train = int((1-test_ratio)*len(self._data_arrays))
                      

        arrays_train = self._data_arrays[:n_train]
        arrays_test = self._data_arrays[n_train:]   
        
        Xtrain, Ytrain = prepare_x_y( arrays_train ,
                                                 skip_steps = skip_steps,
                                                 store_steps_ahead = store_steps_ahead)
                      
        self.train_dataset = SimDataset(Xtrain,Ytrain)
                      
                      
        Xtest, Ytest = prepare_x_y( arrays_test ,
                                               skip_steps = skip_steps, 
                                               store_steps_ahead = store_steps_ahead)
                      
        self.test_dataset = SimDataset(Xtest,Ytest)
                      
        self.test_simulations = self._select_test_simulations(arrays_test)
                      
    def _select_test_simulations(self , arrays_test):
                      
        if len(arrays_test)<self._n_test_simulation:
            n_test_simulation = len(arrays_test)
        else:
            n_test_simulation = self._n_test_simulation
        
        test_simulations = [arrays_test[index][8:] for index in np.arange(0,n_test_simulation,1)]
        
        extra_sims = [arrays_test[0],arrays_test[1][5:],arrays_test[2][15:]] #different u0s
        
        test_simulations.extend(extra_sims)
                          
        return test_simulations
        
    def train_dataloader(self):
                      
        return DataLoader(self.train_dataset, batch_size = 60)
                      
    def val_dataloader(self):
                      
        return DataLoader(self.test_dataset, batch_size = 60)
                      
    def plot_simulation(self, simulation):
                      
        
        
        splits = 6
        for i in range(6):
                      
            plt.figure()
            ind = i*int(np.floor((len(simulation)/6)))
                      
            sim = simulation[ind][0]
            plt.imshow(sim)
     
    
#########EVALUATION UTILS


                    
class SimEvalCallback(Callback):
    
    def __init__(self, datamodule,  results_dir, save_every = 1):
        
        self._datamodule = datamodule
        self._results_dir = results_dir
        self._save_every = save_every

        
    def on_epoch_end(self,trainer, model):
        
        epoch = trainer.current_epoch
        
        if epoch%self._save_every == 0:
                
            evaluate_model(model, self._datamodule, self._results_dir, epoch = epoch)
        
def evaluate_model(model, datamodule, results_dir, epoch = ""):
    
    try:
        os.makedirs(results_dir)
    except:
        pass
    
    skip = datamodule._skip_steps
    
    for i,test_sim in enumerate( datamodule.test_simulations):
        
        if epoch:
            name = "epoch_{}_sim_{}.gif".format(epoch,i)
        else:
            name = "sim_{}.gif".format(i)
            
        name = os.path.join(results_dir, name)
        
        test_sim = test_sim[::skip]
        
        pred_sim, real_sim = eval_sim(model, test_sim)

        make_simulation_gif(pred_sim, real_sim, name, duration = 1, skip_time = skip)
        
    
        plot_phases(pred_sim, real_sim, results_dir, index = i,epoch = epoch)