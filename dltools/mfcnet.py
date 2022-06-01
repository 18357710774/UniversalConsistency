# init_type: 0: zero-init, 1: Orth-Unif, 2: Orth-Gauss, 3: LeCun-Unif, 4: LeCun-Gauss, 5: Xavier, 6: MSRA
# act_type, 1: sigmoid; 2: relu; 3: leaky_relu; 4: linear; 5: tanh;

import torch
import torch.nn as nn
from torch.nn import init

class FCNet(nn.Module):
    def __init__(self, input_dim, output_dim=1,
                 n_hiddens=None, init_type='xavier_uniform',
                 act_type='relu', BN=False, dropout=False):                # 定义网络，设置需要传入的一些参数，默认都为False
        super(FCNet,self).__init__()
        if n_hiddens is None:  # 隐藏层的神经元个数
            n_hiddens = [10, 10, 10]
        self.do_bn = BN
        self.do_drop = dropout                                   #根据传入网络中的参数来决定是否执行dropout或者batch normalization
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons = [input_dim] + n_hiddens + [output_dim]

        self.layer_list = []
        for i in range(len(self.neurons)-1):
            fc_layer = nn.Linear(self.neurons[i], self.neurons[i+1])
            self.layer_list.append(fc_layer)
            if i < len(self.neurons)-2:                          # 最后一层不需要激活函数
                act_layer = self.act_layer(act_type)
                self.layer_list.append(act_layer)
            if self.do_bn:
                bn = nn.BatchNorm1d(self.neurons[i], momentum=0.5)
                self.layer_list.append(bn)
            if self.do_drop:
                drop = nn.Dropout(0.5)
                self.layer_list.append(drop)
        # for i in range(len(self.neurons)-1):
        #     fc = nn.Linear(self.neurons[i], self.neurons[i+1])
        #     setattr(self,'fc%i'%i,fc)                            # setattr函数用来设置属性，其中第一个参数为继承的类别，第二个为名称，第三个是数值
        #     self.fcs.append(fc)
        #     self.__set__init(fc)                                 # 初始化网络训练参数
        #     if self.dobn:
        #         bn = nn.BatchNorm1d(in_putsize, momentum=0.5)
        #         setattr(self,'bn%i'%i,bn)
        #         self.bns.append(bn)
        #     if self.dodrop:
        #         drop = nn.Dropout(0.5)
        #         setattr(self,'drop%i'%i,drop)
        #         self.drops.append(drop)

        self.regressor = nn.Sequential(*self.layer_list)
        self.weight_init(self.regressor, init_type, act_type)


    def weight_init(self, net, init_type, act_type):
        for layer in net:
            if isinstance(layer, nn.Linear):
                self.__set__init(layer, init_type, act_type)


    def __set__init(self, layer, init_type='xavier_uniform', act_type='relu'):
        # nn.init.normal_(layer.weight.data, mean=0, std=0.01)
        # nn.init.constant_(layer.bias, 0.0)
        if init_type == 'zero':
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'orthogonal_uniform':
            self.OrthogonalUniform(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'orthogonal_normal':
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'lecun_uniform':
            self.LecunUniform(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'lecun_normal':
            self.LecunNormal(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(act_type))
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain(act_type))
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity=act_type)
            nn.init.constant_(layer.bias, 0.0)
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=act_type)
            nn.init.constant_(layer.bias, 0.0)
        else:
            raise ValueError("Unsupported init_type {}".format(init_type))


    def LecunNormal(self, tensor):
        r"""Lecun normal initializer.
          Initializers allow you to pre-specify an initialization strategy, encoded in
          the Initializer object, without knowing the shape and dtype of the variable
          being initialized.
          Draws samples from a truncated normal distribution centered on 0 with `stddev
          = sqrt(1 / fan_in)` where `fan_in` is the number of input units in the weight
          tensor.
          References:
            - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
        """
        fan = _calculate_fan_in_and_fan_out(tensor)
        std = 1.0 / math.sqrt(fan)
        with torch.no_grad():
            return tensor.normal_(0., std)

    def LecunUniform(self, tensor):
        r"""Lecun uniform initializer.
          Draws samples from a uniform distribution within `[-limit, limit]`,
          where `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
          weight tensor).
          References:
            - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
          """
        fan = _calculate_fan_in_and_fan_out(tensor)
        bound = math.sqrt(3.0) / math.sqrt(fan)
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def OrthogonalUniform(tensor, gain=1):
        r"""Fills the input `Tensor` with a (semi) orthogonal matrix, as
        described in `Exact solutions to the nonlinear dynamics of learning in deep
        linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
        at least 2 dimensions, and for tensors with more than 2 dimensions the
        trailing dimensions are flattened.
        """
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")

        rows = tensor.size(0)
        cols = tensor.numel() // rows
        flattened = tensor.new(rows, cols).uniform_(-1, 1)
        if rows < cols:
            flattened.t_()
        # Compute the qr factorization
        q, r = torch.qr(flattened)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        if rows < cols:
            q.t_()

        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(gain)
        return tensor

    def act_layer(self, act_type='relu'):
        if act_type == 'sigmoid':
            return nn.Sigmoid()
        elif act_type == 'relu':
            return nn.ReLU()
        elif act_type == 'leaky_relu':
            return nn.LeakyReLU()
        elif act_type == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError("Unsupported act_type {}".format(act_type))

    def forward(self, x):
        y_pred = self.regressor(x)
        return y_pred


if __name__ == '__main__':
    net = FCNet(input_dim=4)


    print(net)
