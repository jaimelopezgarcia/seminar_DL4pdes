import torch
from torch import nn






class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels , n_layers = 2, stride = 1, padding = 1):
        
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
    
    def __init__(self, in_channels, hidden_channels, out_channels, blocks = [2, 2, 2, 2, 2]):
        
        super().__init__()
        
        layers = []
        
        for i,_block in enumerate(blocks):
            
            
            if i == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels
                
                
            layer = ResidualBlock(_in_channels, hidden_channels, stride = 1, padding=1)
            
            layers.append(layer)
            

            
        self._hidden_layers = nn.ModuleList(layers)
        
        
        self._out_layer = nn.Conv2d( hidden_channels , out_channels, kernel_size=3, stride = 1,
             padding=1, bias=True)
        
        
    def forward(self, x):
        
        _x = x
        
        for layer in self._hidden_layers:
            
            _x = layer(_x)
            
        _x = self._out_layer(_x)
            
        return _x