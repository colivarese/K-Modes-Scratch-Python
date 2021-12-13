import numpy as np
from KModes import K_modes
data = np.asarray([['A','B','A','B','C'],
        ['A','A','A','B','B'],
        ['C','A','B','B','A'],
        ['A','B','B','A','C'],
        ['C','C','C','B','A'],
        ['A','A','A','A','B'],
        ['A','C','A','C','C'],
        ['C','A','B','B','C'],
        ['A','A','B','C','A'],
        ['A','B','B','A','C']])

kmodes = K_modes(2, 10)
centroids = kmodes.fit(data)
print(f'\nLast centroids: \n{centroids}')