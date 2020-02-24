import numpy as np

weibull_units = np.load("/home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/PGCR-Results-Analysis/weibull_units.npy")
beta_units = np.load("/home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/PGCR-Results-Analysis/beta_units.npy")

print(weibull_units == beta_units)