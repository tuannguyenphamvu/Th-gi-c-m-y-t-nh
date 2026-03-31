import matplotlib.pyplot as plt 
import numpy as np

N = 500 #number if pixels 
cx,cy = N//2,  N//2
R = N//4  
x = np.linspace(0,N, N*20)
y = np.linspace(0,N, N*20)
image = np.zeros((N, N), dtype=np.uint8) 
for i in range(N):
    image[i,i] = 255
    image[i, N-i-1] = 255
for _x in x:
    for _y in y:
        if np.abs((_x - cx)**2 + (_y - cy)**2 -  R**2) <=0.5:
            image[int(_x), int(_y)] = 255
plt.imshow(image, cmap='gray')

# print(image)
# plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
