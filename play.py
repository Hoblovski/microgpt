import numpy as np
import numpy.random as npr

#       np.random.uniform(low, high, size)
#
#       x.shape
#       x.mean()
#       x.std()

def rmsnorm(x):
    ms = np.mean(1e-5 + x**2)
    scale = ms ** -0.5
    return scale * x

def testrmsnorm():
    print('='*78)
    print('test rmsnorm')
    lrs=[ 
         [-1, 1, 4], 
         [-2, 2, 10], 
         [-100, 100, 40],
         [300, 600, 60]
         ]
    for l, r, shape in lrs:
        x = npr.uniform(l, r, shape)
        print(f'{l=}, {r=}, {shape=}')
        print(f'  {x.mean()=:<20.2f}, {x.std()=:<20.2f}')
        y = rmsnorm(x)
        print(f'  {y.mean()=:<20.2f}, {y.std()=:<20.2f}')
        print(f'  {(y**2).mean()=:<20.2f}')
    # RMSnorm 后保证 mean y**2 = 1
    print('='*78)

testrmsnorm()


