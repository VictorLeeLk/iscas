# pytorch笔记

## 1、torch.nn.Linear()

`torch.nn.``Linear`(*in_features*, *out_features*, *bias=True*)[[SOURCE\]](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear)

Applies a linear transformation to the incoming data: $y = xA^T + b$

- Parameters

  **in_features** – size of each input sample

  **out_features** – size of each output sample

  **bias** – If set to `False`, the layer will not learn an additive bias. Default: `True`

- Shape:

  Input: (*N*,∗,$H_{in}$) where ∗ means any number of additional dimensions and $H_{in}$  =in_features

  Output: (N,*, $H_{out}$)where all but the last dimension are the same shape as the input and *$H_{out}$*=out_features .

- Variables

  **~Linear.weight** – the learnable weights of the module of shape (out_features,in_features) . The values are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ , where$k=\frac{1} {in_features}$ 

  **~Linear.bias** – the learnable bias of the module of shape (out_features) . If `bias` is `True`, the values are initialized from  $\mathcal{U}(-\sqrt{k}, \sqrt{k})$where $k = \frac{1}{\text{in_features}}$

Examples:

```
>>> m = nn.Linear(20, 30)
   #m.weights.data.size=(30,20)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 30])
```

源码：

F:torch.nn.Functional ,该类的方法Linear会对权值矩阵先求转置

```
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
```

