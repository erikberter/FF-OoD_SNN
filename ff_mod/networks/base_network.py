import torch
from torch.optim import Adam
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sigmoid


from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet, AbstractForwardForwardLayer

from ff_mod.overlay import Overlay

class AnalogNetwork(AbstractForwardForwardNet):

    def __init__(
            self,
            overlay_function : Overlay,
            dims,
            loss_function,
            learning_rate = 0.001,
            internal_epoch = 20,
            first_prediction_layer = 0,
            save_activity = False,
            is_symmetric = False
        ):
        
        super().__init__(overlay_function, first_prediction_layer, False)
        
        self.internal_epoch = internal_epoch
        
        for d in range(len(dims) - 1):
            layer = Layer(dims[d], dims[d + 1], loss_function, learning_rate, save_activity, internal_epoch, is_symmetric)
            self.layers.append(layer)

        self.layers[0].is_input_layer = True
        
    def adjust_data(self, data):
        return data

class Layer(AbstractForwardForwardLayer):
    def __init__(
            self,
            in_features,
            out_features,
            loss_function,
            learning_rate = 0.001,
            save_activity = False,
            internal_epoch = 10,
            is_symmetric = False,
            **kwargs
        ):
        super().__init__(in_features, out_features, loss_function = loss_function, save_activity = save_activity, **kwargs)
        
        self.is_input_layer = False
        
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(in_features)
        
        self.opt = Adam(self.parameters(), lr=learning_rate)
        
        self.is_symmetric = is_symmetric

        self.num_epochs = internal_epoch

    def get_goodness(self, x):
        if not self.is_symmetric:
            goodness= x.pow(2).mean(1)
            return goodness
        else:
            middle_part = x.shape[1]//2
            rest = x[:,:middle_part].pow(2).sum(1) / (x.pow(2).sum(1) + 0.0001)

            return rest
    
    def forward(self, x, positivity_type = None):
        
        bias_term = 0
        if self.bias is not None:
            bias_term = self.bias.unsqueeze(0)

        result =  self.relu(torch.mm(x, self.weight.T) + bias_term)
            
        self.add_activity(result, positivity_type)
        
        return result
    
    def __call__(self, x, positivity_type=None):
        return self.forward(x, positivity_type)



class Conv2DLayer(AbstractForwardForwardLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            loss_function,
            learning_rate=0.001,
            save_activity=False,
            internal_epoch=10,
            stride=1,
            padding=0,
            **kwargs
        ):
        super().__init__(in_channels, out_channels, loss_function=loss_function, save_activity=save_activity, **kwargs)

        self.is_input_layer = False

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.bn = BatchNorm2d(in_channels, affine=False)

        self.relu = ReLU()

        self.opt = Adam(self.parameters(), lr=learning_rate)

        self.dropout = torch.nn.Dropout(0.5)
        
        self.num_epochs = internal_epoch

    def get_goodness(self, x):
        return x.pow(2).mean(dim=(1, 2, 3))
        # x shape is (batch_size, out_channels, height, width)
        mean_channel = x.shape[1]//2
        pos_g = x[:,:mean_channel,:,:].pow(2).mean(dim=(1, 2, 3))
        neg_g = x[:,mean_channel:,:,:].pow(2).mean(dim=(1, 2, 3))
        
        return pos_g / (pos_g + neg_g + 0.0001)

    def forward(self, x, positivity_type=None):
        #result = self.relu(self.conv(self.bn(x)))
        result = self.relu(self.conv(x))
        
        result = self.dropout(result)

        self.add_activity(result, positivity_type)

        return result

    def __call__(self, x, positivity_type=None):
        return self.forward(x, positivity_type)