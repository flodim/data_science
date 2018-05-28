import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors


# read data from csv file
titanic_data = np.genfromtxt('titanic.dat', delimiter=',', skip_header=1)

# extract data
classes = titanic_data[:, 0]
ages = titanic_data[:, 1]
sexes = titanic_data[:, 2]
survived = titanic_data[:, 3]

for i in ages:
    print(i)

# 3d scatter plot
fig_3d = plt.figure(1)
ax = Axes3D(fig_3d)
ax.scatter(classes, ages, sexes)
ax.set_xlabel('class')
ax.set_ylabel('age')
ax.set_zlabel('sex')
ax.set_title('class vs age vs sex')
fig_3d.show()

gamma = 2

# class vs age
fig_class_age = plt.figure(2)
plt.hist2d(classes,ages)
plt.xlabel('class')
plt.ylabel('age')
plt.title('class vs age')
fig_class_age.show()

# class vs sex
fig_class_sex = plt.figure(3)
plt.hist2d(classes, sexes)
plt.xlabel('class')
plt.ylabel('sex')
plt.title('class vs sex')
fig_class_sex.show()

# age vs sex
fig_age_sex = plt.figure(4)
plt.hist2d(ages, sexes)
plt.xlabel('age')
plt.ylabel('sex')
plt.title('age vs sex')
fig_age_sex.show()


