from codecs import replace_errors
import logging
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict

class StackedModel(nn.Module):
    def __init__(self, models, num_clients=None, debug=False):
        super().__init__()

        # If only one model is passed, we duplicate it
        if num_clients is not None:
            assert isinstance(models, nn.Module) and num_clients >= 1
            models = [models] * num_clients
            
        self.num_clients = len(models)
        self._net = deepcopy(models[0])
        self._stack_layers(self._net, models)

        if debug:
            self._check_output(models)

    @torch.no_grad()
    def _check_output(self, models):
        logging.info('checking stacked model output')
        dev = next(models[0].parameters()).device
        self.to(dev)
        xs = [torch.randn(16, 3, 32, 32) for _ in range(10)]
        xs = [x.to(dev) for x in xs]
        self._net.eval()
        out = self._net(torch.cat(xs))
        expected = []
        for x, model in zip(xs, models):
            model.eval()
            expected += [model(x)]

        expected = torch.cat(expected)
        assert torch.allclose(out, expected, atol=1e-5)
        logging.info('stacked model and individual models align.')


    def forward(self, *args, **kwargs):
        return self._net(*args, **kwargs)
        
    def _stack_layers(self, base_model, models):
        """
        modify `base_model` to contain multiple models
        """
        router = {
            nn.Linear : StackedLinear, 
            nn.Conv2d : StackedConv2d, 
            nn.BatchNorm2d : BatchNorm2dWrapper,
            nn.GroupNorm: GroupNormWrapper,
        }

        def replace_layers(parent_mod, prefix=None):
            for name, mod in parent_mod.named_children():
                sub_name = name if prefix is None else f'{prefix}.{name}'
                if len([x for x in mod.children()]) > 0: 
                    replace_layers(mod, sub_name)

                if type(mod) in router.keys():
                    all_mods = [rec_getattr(model, sub_name) for model in models] 
                    new_mod = router[type(mod)](all_mods)
                    rec_setattr(base_model, sub_name, new_mod)
                else:
                    # TODO (Lucas): make sure this is robust
                    # To ensure that no methods with params are being missed
                    import pdb    
                    assert all('.' in k for k in mod.state_dict().keys()), pdb.set_trace()

        replace_layers(base_model)
        
    def extract_models(self, avg=False):
        # return a `state_dict` for each client
        if avg: 
            dicts = OrderedDict()
        else:
            dicts = [OrderedDict() for _ in range(self.num_clients)]
        for key, value in self._net.state_dict().items():
            if value.ndim == 0: 
                # typically counters
                values = [value] * self.num_clients
            else:
                values = [x.squeeze(0) for x in value.chunk(self.num_clients)]
            if avg:
                dicts[key] = sum(values) / self.num_clients
            else:
                for d, value in zip(dicts, values): 
                    d[key] = value
    
        return dicts


# --- Operators 
class StackedLinear(nn.Module):
    def __init__(self, conv_layers):
        super().__init__()
        self.num_clients = len(conv_layers)
        self.in_features = conv_layers[0].in_features
        self.out_features = conv_layers[0].out_features

        self.weight = nn.Parameter(torch.stack([layer.weight for layer in conv_layers]))

        bias = conv_layers[0].bias
        if bias is not None:
            self.bias =  nn.Parameter(torch.stack([layer.bias for layer in conv_layers]))
        else:
            self.bias = None

    def forward(self, input):
        # input is (num_clients *  batch_size, in_features)
        input = input.reshape(self.num_clients, -1, self.in_features)
        out = torch.einsum('cbi,coi->cbo', (input, self.weight))
        if self.bias is not None: 
            out += self.bias.unsqueeze(1)

        return out.view(-1, self.out_features)

class StackedConv2d(nn.Module): 
    def __init__(self, conv_layers):
        super().__init__()
        self.num_clients = len(conv_layers)

        for name, value in vars(conv_layers[0]).items():
            if not name.startswith('_'):
                setattr(self, name, value)

        assert self.groups == 1, 'Not supported for now.' 

        # out_c, in_c, ks[0], ks[0]
        W = torch.stack([layer.weight for layer in conv_layers])
        self.weight = nn.Parameter(W)

        if conv_layers[0].bias is not None:
            self.bias =  nn.Parameter(torch.stack([layer.bias for layer in conv_layers]))
        else:
            self.bias = None

    def forward(self, input):
        # input is (num_clients *  batch_size, in_c, H, W)
        input = input.view(self.num_clients, -1, *input.shape[1:])
        num_clients, bs = input.shape[:2]

        # make it batch_size, num_clients * in_c, H, W 
        input = input.transpose(0,1).flatten(1, 2)

        # num_clients, O, I, kH, kW --> num_clients * O, I, kH, kW
        W = self.weight.flatten(0,1)

        bias = None if self.bias is None else self.bias.flatten()

        out = F.conv2d(input, W, bias, self.stride, self.padding, self.dilation, self.num_clients)
        out = out.reshape(bs, num_clients, -1, *out.shape[-2:])
        out = out.transpose(0,1)

        return out.reshape(-1, *out.shape[2:])

# --- Normalizations
class BatchNorm2dWrapper(nn.BatchNorm2d):
    def __init__(self, bn_layers):
        self.num_clients = len(bn_layers)
        
        dev = None if not bn_layers[0].affine else bn_layers[0].weight.device
        super().__init__(
            bn_layers[0].num_features * self.num_clients, 
            eps=bn_layers[0].eps, 
            affine=bn_layers[0].affine, 
            momentum=bn_layers[0].momentum, 
            track_running_stats=bn_layers[0].affine,
            device=dev 
        )

        # fill the buffers with the appropriate values
        if self.track_running_stats:
            mu  = torch.cat([bn.running_mean for bn in bn_layers])
            var = torch.cat([bn.running_var  for bn in bn_layers])
            self.running_mean.copy_(mu)
            self.running_var.copy_(var)

        if self.affine: 
            self.weight.data.copy_(torch.cat([bn.weight for bn in bn_layers]))
            self.bias.data.copy_(torch.cat([bn.bias for bn in bn_layers]))

    def forward(self, x):
        x = x.view(self.num_clients, -1, *x.shape[1:])
        client_bs = x.size(1)
        x = x.transpose(0,1).reshape(client_bs, -1, *x.shape[3:])
        out =  F.batch_norm(
            x, 
            self.running_mean, 
            self.running_var, 
            self.weight, 
            self.bias, 
            self.training, 
            self.momentum, 
            self.eps
        )
        out = out.reshape(client_bs, self.num_clients, -1, *out.shape[2:])
        out = out.transpose(0,1).flatten(0,1)
        return out

class GroupNormWrapper(nn.GroupNorm):
    def __init__(self, gn_layers):
        self.num_clients = len(gn_layers)

        dev = None if not gn_layers[0].affine else gn_layers[0].weight.device
        super().__init__(
            gn_layers[0].num_groups * self.num_clients, 
            gn_layers[0].num_channels * self.num_clients, 
            affine=gn_layers[0].affine,
            device=dev
        )

        if self.affine: 
            self.weight.data.copy_(torch.cat([gn.weight for gn in gn_layers]))
            self.bias.data.copy_(torch.cat([gn.bias for gn in gn_layers]))

    def forward(self, x):
        x = x.view(self.num_clients, -1, *x.shape[1:])
        client_bs = x.size(1)
        x = x.transpose(0,1).reshape(client_bs, -1, *x.shape[3:])
        out =  F.group_norm(
            x, 
            self.num_groups,
            weight = self.weight, 
            bias = self.bias
        )
        out = out.reshape(client_bs, self.num_clients, -1, *out.shape[2:])
        out = out.transpose(0,1).flatten(0,1)
        return out


def rec_getattr(obj, path):
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj

def rec_setattr(obj, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

def dict_allclose(dicts, ok=['num_batches_tracked']):
    for key, value in dicts[0].items():
        if any(ex in key for ex in ok):
            # print(f'skipping {key}')
            continue
        values = [d[key] for d in dicts[1:]]
        if not all(torch.allclose(v, value) for v in values): 
            import pdb; pdb.set_trace()
            return False
    return True


if __name__ == '__main__':
    class args: 
        pass
    args.num_classes = 10
    args.num_channels = 3
    args.norm = 'group_norm'

    C = 2
    from models import get_model
    models = [get_model('resnet18')(args) for _ in range(C)]
    
    # pass some data so the bn buffers are not id map
    for model in models:
        model.train()
        for _ in range(3):
            x = torch.randn(16, 3, 32, 32)
            model(x)

    for trial in range(10):
        print(trial)
        xs = [torch.randn(16, 3, 32, 32) for _ in range(C)]
        new_model = StackedModel(models)
        expected = torch.cat([model(x) for model, x in zip(models, xs)])

        value = new_model(torch.cat(xs))

        torch.allclose(expected, value, atol=1e-6)

        model_dicts = new_model.extract_models()
        for model, model_dict in zip(models, model_dicts):
            assert dict_allclose([model.state_dict(), model_dict])

    print(model)
    print('model output matches concatenation of individual models.')

    # Test out individual components    
    # --- GN 
    gns = [nn.GroupNorm(10, 100) for _ in range(5)]
    for bn in gns:
        bn.eval()
        bn.weight.data.uniform_(-2, 2)
        bn.bias.data.uniform_(-2, 2)

    xs = [torch.randn(16, 100, 24, 24) for _ in range(5)]
    expected = torch.cat([l(x) for l, x in zip(gns, xs)])

    all_bns = GroupNormWrapper(gns)
    all_bns.eval()
    value = all_bns(torch.cat(xs))
    assert torch.allclose(expected, value)

    # --- BN 
    bns = [nn.BatchNorm2d(10) for _ in range(5)]
    for bn in bns:
        bn.eval()
        bn.running_mean.data.uniform_(-2, 2)
        bn.running_var.data.uniform_(1e-3, 2)
        bn.weight.data.uniform_(-2, 2)

    xs = [torch.randn(16, 10, 24, 24) for _ in range(5)]
    expected = torch.cat([l(x) for l, x in zip(bns, xs)])

    all_bns = BatchNorm2dWrapper(bns)
    all_bns.eval()
    value = all_bns(torch.cat(xs))
    assert torch.allclose(expected, value)

    # --- Conv2d
    convs = [nn.Conv2d(10, 20, kernel_size=3) for _ in range(5)]
    # n_clients, bs, C, H, W
    xs = [torch.randn(16, 10, 24, 24) for _ in range(5)]

    expected = torch.cat([l(x) for l, x in zip(convs, xs)])
    all_convs = StackedConv2d(convs)
    value = all_convs(torch.cat(xs))

    assert torch.allclose(expected, value, atol=1e-6)

    # --- Linear
    linears = [nn.Linear(10, 20) for _ in range(5)]
    xs = [torch.randn(5, 10) for _ in range(5)]

    expected = torch.cat([l(x) for l, x in zip(linears, xs)])
    all_linear = StackedLinear(linears)
    value = all_linear(torch.cat(xs))

    assert torch.allclose(expected, value)

