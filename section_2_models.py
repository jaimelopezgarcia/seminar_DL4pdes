import os
import torch
from torch import nn
import skimage.transform
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
from torch.autograd import grad






class BasicNet(torch.nn.Module):
    
    def __init__(self, n_inputs, n_outputs, hidden_layers = 5, neurons_hidden = 50, activation = "sin"):
        super().__init__()
        
        self._neurons_hidden = neurons_hidden
        self._hidden_layers = hidden_layers
        
        layers = []
        for i in range(self._hidden_layers):
            
            if i == 0:
                _in = n_inputs
            else:
                _in = self._neurons_hidden
            
            layer = torch.nn.Linear(_in, self._neurons_hidden)
            
            layers.append(layer)
            
        self.hidden_layers = torch.nn.ModuleList(layers)
        
        self._out_layer = torch.nn.Linear(self._neurons_hidden, n_outputs)
        
        if activation == "sin":
            self._activation = torch.sin
        elif activation == "tanh":
            self._activation = torch.tanh
        
    def forward(self, x):

        X = x
        
        for i,hlayer in enumerate(self.hidden_layers):
            
            if i == 0:
                w = 3
            else:
                w = 1
                
            X = self._activation( w*hlayer(X) )
            
        u = self._out_layer(X)
        
        
        
        return u
   


class SineLayer(nn.Module):

    
    def __init__(self, in_features, out_features, 
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    

    
class Siren(nn.Module):
    def __init__(self, in_features, out_features , hidden_layers = 5 ,  hidden_features = 50,
                 first_omega_0=30, hidden_omega_0=1., final_linear = False):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        if final_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                                np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)

        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        
        output = self.net(coords)
        
        return output 
    
    
    
    
    
    
    
###  POISSON 
 
def Nu_poisson_1D(u_pred, x, source_fun, sigma):
    
    source = torch.Tensor( source_fun(x.clone().detach().numpy().reshape(-1,1)) )

    u_x = grad(u_pred,x, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0]
    u_xx = grad(u_x, x, create_graph = True, grad_outputs = torch.ones_like(u_x))[0]
    

    f = source + sigma*u_xx
    
    return f.view(-1,1)


def Nu_poisson_2D(u_pred, X , source_fun, sigma):
    
    x,y = X[:,0], X[:,1]
    source = torch.Tensor( source_fun(x.clone().detach().numpy().reshape(-1,1), y.clone().detach().numpy().reshape(-1,1)) )

    u_x = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0].view(-1,1)
    u_xx = grad(u_x, X, create_graph = True, grad_outputs = torch.ones_like(u_x))[0][:,0].view(-1,1)
    
    u_y = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0].view(-1,1)
    u_yy = grad(u_y, X, create_graph = True, grad_outputs = torch.ones_like(u_y))[0][:,1].view(-1,1)

    f = source + sigma*(u_yy+u_xx)
    
    return f.view(-1,1)
    
    
def Nu_AC1D(u_pred, X , eps, M):
    

    u_x = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0].view(-1,2)
    u_xx = grad(u_x, X, create_graph = True, grad_outputs = torch.ones_like(u_x))[0][:,0].view(-1,1)
    
    u_t = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0][:,1].view(-1,1)

    f = u_t-M*(u_xx-(1/eps**2)*(u_pred**2-1)*u_pred)
    
    return f.view(-1,1) 
    
    
class Poisson1D_Dataset(Dataset):
    def __init__(self, X):
        #self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self._X = torch.Tensor(X).view(-1,1)


    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):

        sample = self._X[idx]
        
        return sample

    
    
    
class Poisson1DModel(pl.LightningModule):
    
    def __init__(self,  model, source_fun, sigma, validation_fun, xa = -1, xb = 1, Npoints = 70, shuffle = False):
        super().__init__()
        
        self._model = model

        
        self._source_fun = source_fun
        
        self._xa = xa
        self._xb = xb
        self._Npoints = Npoints
        self._sigma = sigma
        self._fun_f = source_fun
        self._fun_u = validation_fun
        
        Xt = torch.Tensor(np.linspace(xa,xb, Npoints)).view(-1,1)


        self._ua = torch.Tensor([self._fun_u(xa)]).view(-1,1)
        self._ub = torch.Tensor([self._fun_u(xb)]).view(-1,1)

        self._Xa = torch.Tensor([xa]).view(-1,1)
        self._Xb = torch.Tensor([xb]).view(-1,1)
        
        self._train_dataloader = DataLoader( Poisson1D_Dataset(Xt), batch_size = len(Xt), shuffle =  shuffle)
        self._test_dataloader = DataLoader( Poisson1D_Dataset(Xt), batch_size = len(Xt), shuffle =  shuffle)
       
    def get_train_dataloader(self):
        return self._train_dataloader
    
    def get_test_dataloader(self):
        return self._test_dataloader

    def forward(self,x):
        
        x = self._model(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        
       
        fun_f = self._fun_f
        sigma = self._sigma
        
        Xa, Xb, ua, ub = self._Xa, self._Xb, self._ua, self._ub

        Xt = batch.clone().requires_grad_(True)
        
        u_int = self(Xt)

        residual = Nu_poisson_1D(u_int, Xt, fun_f, sigma)   

        loss_int = torch.mean(torch.square(residual))

        loss_b = torch.mean(  torch.square(self(Xa)-ua) +torch.square(self(Xb)-ub))

        loss = loss_int + loss_b   
        
        self.log("loss_int", loss_int, on_epoch = True, prog_bar = True)
        self.log("loss_b", loss_b, on_epoch = True, prog_bar = True)
        self.log("loss", loss, on_epoch = True, prog_bar = True)
        

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        fun_u = self._fun_u
        
        Xt = batch
        real = torch.Tensor( fun_u(Xt.clone().detach().numpy()) ).view(-1,1)
        pred = self(Xt)
        
        loss_val = torch.mean( torch.square(pred - real))
        
        self.log("loss_val", loss_val, on_epoch = True, prog_bar = True)
        
        return loss_val

    
        
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold = 1e-6 ,verbose = True, eps = 1e-7)
        
        """
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'loss_val'
        }
        """
        
        
        return optimizer
    
    
    
class Poisson2D_Dataset(Dataset):
    def __init__(self, X):
        #self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self._X = torch.Tensor(X).view(-1,2)


    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):

        sample = self._X[idx]
        
        return sample

    
    
    
    
class Poisson2DModel(pl.LightningModule):
    
    def __init__(self, model, source_fun, sigma, validation_fun, xa = -1, xb = 1, Npoints = 70, shuffle = False, batch_size = None):
        super().__init__()
        
        self._model = model

        
        
        self._xa = xa
        self._xb = xb
        self._Npoints = Npoints
        self._sigma = sigma
        self._fun_f = source_fun
        self._fun_u = validation_fun
        
        fun_u = self._fun_u
        fun_f = self._fun_f
        


        Xnp ,Ynp = np.linspace(xa,xb,Npoints), np.linspace(xa,xb,Npoints)
        Xm,Ym = np.meshgrid(Xnp,Ynp)
        Xm = Xm.reshape(-1,1)
        Ym = Ym.reshape(-1,1)

        Xall = np.concatenate((Xm,Ym), axis = 1)


        Xt = torch.Tensor(Xnp)
        Yt = torch.Tensor(Ynp)
        Xall = torch.Tensor(Xall)

        
        real = torch.Tensor(fun_u(Xall[:,0].clone().detach().numpy(), Xall[:,1].clone().detach().numpy())).view(-1,1)

        zeros = torch.zeros((Xt.shape[0],2))

        Xall_left , Xall_right, Xall_top, Xall_bottom = zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()

        Xall_left[:,0], Xall_right[:,0] = xa, xb
        Xall_left[:,1], Xall_right[:,1] = Yt, Yt

        Xall_bottom[:,1], Xall_top[:,1] = xa, xb
        Xall_bottom[:,0], Xall_top[:,0] = Xt, Xt



        u_left, u_right, u_bottom, u_top = self._eval_bound(fun_u,Xall_left), self._eval_bound(fun_u,Xall_right), self._eval_bound(fun_u,Xall_bottom), self._eval_bound(fun_u,Xall_top)

        
        self._Xall_left, self._Xall_right, self._Xall_bottom, self._Xall_top = Xall_left, Xall_right, Xall_bottom, Xall_top
        self._u_left, self._u_right, self._u_bottom, self._u_top = u_left, u_right, u_bottom, u_top
        
        if not(batch_size):
            self._batch_size = len(Xall)
        else:
            self._batch_size = batch_size

        self._train_dataloader = DataLoader( Poisson2D_Dataset(Xall), batch_size = self._batch_size, shuffle =  shuffle)
        self._test_dataloader = DataLoader( Poisson2D_Dataset(Xall), batch_size = self._batch_size, shuffle =  shuffle)
       
    def _eval_bound(self,fun, X):
        return torch.Tensor(fun(X[:,0].clone().detach().numpy(), X[:,1].clone().detach().numpy())).view(-1,1)
    
    def get_train_dataloader(self):
        return self._train_dataloader
    
    def get_test_dataloader(self):
        return self._test_dataloader

    def forward(self,x):
        
        x = self._model(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        
       
        fun_f = self._fun_f
        sigma = self._sigma
        
        Xall_left, Xall_right, Xall_bottom, Xall_top = self._Xall_left, self._Xall_right, self._Xall_bottom, self._Xall_top
        u_left, u_right, u_bottom, u_top = self._u_left, self._u_right, self._u_bottom, self._u_top

        Xall = batch.clone().requires_grad_(True)
        
        u_int = self(Xall)


        residual = Nu_poisson_2D(u_int, Xall , fun_f, sigma)   

        loss_int = torch.mean(torch.square(residual))

        loss_b = torch.mean(  torch.square(self(Xall_left)-u_left) +torch.square(self(Xall_right)-u_right) +torch.square(self(Xall_bottom)-u_bottom) +torch.square(self(Xall_top)-u_top))

        loss = loss_int + loss_b   
    
    
        self.log("loss_int", loss_int, on_epoch = True, prog_bar = True, on_step = False)
        self.log("loss_b", loss_b, on_epoch = True, prog_bar = True, on_step = False)
        self.log("loss", loss, on_epoch = True, prog_bar = True)
        

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        fun_u = self._fun_u
        
        Xt = batch
        Xnp = batch.clone().detach()
        real = torch.Tensor( fun_u(Xnp[:,0],Xnp[:,1] ) ).view(-1,1)
        pred = self(Xt)
        
        loss_val = torch.mean( torch.square(pred - real))
        
        self.log("loss_val", loss_val, on_epoch = True, prog_bar = True, on_step = False)
        
        return loss_val

    
        
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold = 1e-6 ,verbose = True, eps = 1e-7)
        
        """
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'loss_val'
        }
        """
        
        
        return optimizer
    
    
from utils import plot_2D_comparison_analytical,fig_to_array, make_gif
   
class SimEvalCallback(Callback):

    def __init__(self, val_fun, results_dir, plot_every = 10, save_every = 50, size = (300,300), name = "results"):

        self._results_dir = results_dir
        self._save_every = save_every
        self._epoch = 0
        self._arrays = []
        self._plot_every = plot_every
        self._val_fun = val_fun
        self._size = size
        self._name = name
    def on_epoch_end(self,trainer, model):

        #epoch = trainer.current_epoch
        self._epoch+=1
        epoch = int(self._epoch/2) #dirty fix
        
        
        if epoch%self._plot_every == 0:
            plt.close("all")
            
            fig = plot_2D_comparison_analytical(model, self._val_fun, title = "epoch {}".format(epoch)) 

            array = fig_to_array(fig)
            
            self._arrays.append(array)
            
        if epoch%self._save_every == 0:
            
            make_gif(self._arrays,self._results_dir+"/{}.gif".format(self._name), size = self._size, duration = 0.5)