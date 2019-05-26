import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
x = np.arange(0,1.1,0.01)
fig = plt.figure()

plt1 = fig.add_subplot(121)
plt2 = fig.add_subplot(122)

y1 = 1-(np.exp(x)/np.exp(1.1))**2
# y2 = ((np.exp(1.1))**2 - (np.exp(x))**2)/((np.exp(1.1))**2 + (np.exp(x))**2)
y3 = (np.exp(1.1))**2/((np.exp(1.1))**2 + (np.exp(x))**2)
y4 = (1.1**2)/(1.1**2 + x**2)
y5 = (1.1**2 - x**2)/(1.1**2 + x**2)
plt1.plot(x,y1,color = 'r')
# plt.plot(x,y2,color = 'b')
plt1.plot(x,y3,color = 'g')
plt1.plot(x,y4,color = 'b')
plt1.plot(x,y5,color = 'black')
plt1.set_title('Membership vs distance(<1.5)')



x1 = np.arange(0,3,0.01)

y11 = 1-(np.exp(x1)/np.exp(3))**2
# y2 = ((np.exp(1.1))**2 - (np.exp(x))**2)/((np.exp(1.1))**2 + (np.exp(x))**2)
y31 = (np.exp(3))**2/((np.exp(3))**2 + (np.exp(x1))**2)
y41 = (3**2)/(3**2 + x1**2)
y51 = (3**2 - x1**2)/(3**2 + x1**2)
plt2.plot(x1,y11,color = 'r')
# plt.plot(x,y2,color = 'b')
plt2.plot(x1,y31,color = 'g')
plt2.plot(x1,y41,color = 'b')
plt2.plot(x1,y51,color = 'black')
# plt2.xlabel('Distance')
# plt2.ylabel('Membership')
plt2.set_title('Membership vs distance(>1.5)')
fig.subplots_adjust(hspace=.5,wspace=0.5)
plt.show()
