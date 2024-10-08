import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = -(x-5)**2 + 50
c = np.array(['b' for _ in range(len(x))])
c[y > 40] = 'r'
print(c)
plt.plot(x, y, c=c)
plt.show()
