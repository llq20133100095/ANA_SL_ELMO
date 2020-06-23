import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x1 = np.random.uniform(-10, 10, size=20)
x2 = np.random.uniform(-10, 10, size=20)
# print(x1)
# print(x2)
number = []
x11 = []
x12 = []
for i in range(20):
    number.append(i + 1)
    x11.append(i + 1)
    x12.append(i + 1)
plt.figure(1)
# you can specify the marker size two ways directly:
plt.plot(number, x1, 'bo', markersize=20, label='a')  # blue circle with size 20
plt.plot(number, x2, 'ro', ms=10, label='b')  # ms is just an alias for markersize

lgnd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
lgnd.legendHandles[0]._legmarker.set_markersize(16)
lgnd.legendHandles[1]._legmarker.set_markersize(10)

# fig.subplots_adjust(right=0.8)

plt.show()

fig.savefig('scatter2.png', dpi=600, bbox_inches='tight')

import matplotlib.pyplot as plt
name_list = ['The', 'PRI', 'government', 'crushed', 'a', 'burgeoning', 'student', 'movement', 'with', 'gunfire', 'that',
             'killed', 'scores', 'of', 'peaceful', 'demonstrators']
num_list = [0.13, 0.54, 0.87, 0.9, 0.11, 0.25, 0.48, 0.69, 0.47, 0.88, 0.17, 0.47, 0.37, 0.21, 0.5, 0.4]
plt.xticks(rotation=300)
plt.grid()
plt.ylim([0, 1.0])
plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
plt.savefig('complex.png', dpi=600, bbox_inches='tight')
plt.show()

