import scipy.sparse as sparse
import scipy.stats as stats
import numpy as np
import sys

#file = open('./data/cora_input.dat', 'rb')
#data = np.fromfile(file, dtype=np.float32)
#data.astype('float32').tofile('test')

x = int(sys.argv[1])
y = int(sys.argv[2])
d = float(sys.argv[3])
f = sys.argv[4]

np.random.seed(42)
A = sparse.random(x,y, density=d)
data = (np.array(A.toarray()).flatten())
print(data[0:10])
data.astype('float32').tofile(f+'.dat')
np.savetxt(f+'.txt',np.array(A.toarray()))


