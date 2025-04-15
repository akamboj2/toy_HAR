import torch
import time

timer = time.time()

x = torch.randn(400,3, 1080, 920)

print("Time taken to generate random tensor:", time.time()-timer)

timer = time.time()

y = x[::2,:,:,:] # downsample time by half

print("Time taken to downsample tensor:", time.time()-timer)
timer = time.time()

z = x[::10,:,:,:] # downsample time by 1/10

print("Time taken to downsample tensor:", time.time()-timer)

timer = time.time()

a = x[::10,:,:,:].clone() # downsample time by half

print("Time taken to downsample tensor and clone:", time.time()-timer)

