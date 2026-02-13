import torch

pt_version = torch.__version__
print(f'PyTorch: {pt_version}')

# For PyTorch 2.9+, use torch.amp.autocast without device_type
# For older versions, use torch.cuda.amp.autocast
major, minor = map(int, pt_version.split('.')[:2])

if major >= 2 and minor >= 9:
    print('Using PyTorch 2.9+ API')
    print("autocast(device_type='cuda', dtype=DTYPE) -> autocast('cuda', dtype=DTYPE)")
else:
    print('Using older API')
