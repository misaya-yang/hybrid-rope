import torch
print(f'PyTorch version: {torch.__version__}')
print('Testing autocast...')
try:
    from torch.amp import autocast
    print('torch.amp.autocast available')
except:
    print('torch.amp.autocast not available')
    from torch.cuda.amp import autocast
    print('torch.cuda.amp.autocast available')

# 测试不同的用法
import torch.amp
print('Available in torch.amp:', dir(torch.amp))
