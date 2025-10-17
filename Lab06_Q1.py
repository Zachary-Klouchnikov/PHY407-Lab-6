__authors__ = "Zachary Klouchnikov and Hannah Semple"

# HEADER

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

"""
FUNCTIONS
"""

"""
PART A)
"""
"Constants"
ALPHA = 1.0
BETA = 1.0
V_F = 1.0

"Main Code"
xdot = np.linspace(1, 100, 1000)

f_f = lambda x, da: -(ALPHA + da) * x
f_s = lambda x, db, dv: -(BETA + db) * np.exp(-x / (V_F + dv))

"Plotting Total Absolute Frictional Force"
plt.figure()

# Plotting total absolute frictional force
plt.plot(xdot, np.abs(f_f(xdot, 0.0) + f_s(xdot, 0.0, 0.0)), ls = '-', color = 'Teal', label = r"$\alpha/\beta = 1$")
plt.plot(xdot, np.abs(f_f(xdot, -0.5) + f_s(xdot, 0.5, 0.0)), ls = '-', color = 'Purple', label = r"$\alpha/\beta < 1$")
plt.plot(xdot, np.abs(f_f(xdot, 0.5) + f_s(xdot, -0.5, 0.0)), ls = '-', color = 'Coral', label = r"$\alpha/\beta > 1$")

# Labels
plt.title("Total Absolute Frictional Force", fontsize = 12)
plt.xlabel("Velocity of the Mass (m/s)", fontsize = 12)
plt.ylabel("Force (N)", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()
