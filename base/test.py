import numpy as np
from keras.preprocessing import sequence
a =np.zeros((1,6));
print(a)
a[0, -1] = 2
print(a)
print(a[0,0:-1])
sequence.pad_sequences