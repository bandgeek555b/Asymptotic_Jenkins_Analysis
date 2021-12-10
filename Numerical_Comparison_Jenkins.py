# Allen Mathis
# Fall 2019
# Asymptotic Jenkins
# Numerical Jenkins Element

# This script simulates the original differential equation and the
# asymptotic approximation of the SDOF oscillator with Jenkins element damping.
# This script also plots the solution of the differential equation, the
# associated flow field, as well as the fixed point of the solution.

# Import Libraries
from numpy import *
import math
from math import atan2 as atan2
from math import asin as asin
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

# System Parameters
eps = 0.1
Gamma = 0.92992966#0.3
alpha = -0.19278896690248998#-0.5#1

Omega = 1 + eps*alpha

# Jenkins Parameters
k = 1#0.5
phi = 0.5#0.5

# Linear Damping Parameters
zeta = 0.1#0.5#1

# Numerical Parameters
dt = 1e-3
tmax = 200
deta_1 = eps*dt

# Initial Conditions
x_IC = 0
xdot_IC = 0
z_IC = 0
A0_IC = 0
theta0_IC = 0

# Define the number of time steps in the solution
# This is the same for the numerical and asymptotic solutions,
# as (eps*tmax)/(eps*dt) = tmax/dt
N = math.floor(tmax/dt)

# Allocate memory for the different numerical output variables
# Time
t_state = zeros(N)
# Oscillator displacement and velocity
x_state = zeros(N)
xdot_state = zeros(N)
# Hysteresis internal variable
z_state = zeros(N)
# Asymptotic solution
eta1_state = zeros(N)
xasym_state = zeros(N)
# Asymptotic solution amplitude and phase
A0_state = zeros(N)
theta0_state = zeros(N)
# Cartesian transformation of asymptotic solution
xc_state = zeros(N)
yc_state = zeros(N)
# Linear analytical solution for comparison
x_analytical = zeros(N)

# Assign initial conditions
x_state[0] = x_IC
xdot_state[0] = xdot_IC
z_state[0] = z_IC
A0_state[0] = A0_IC
theta0_state[0] = theta0_IC
xasym_state[0] = A0_IC*sin(theta0_IC)
xc_state[0] = A0_IC*cos(theta0_IC)
yc_state[0] = A0_IC*sin(theta0_IC)

# Define the Fourier components involved in the slow-flow calculations
def a1(A0):
	if k*A0 <= phi:
		output = 2*zeta*A0
	else:
		output = 4*phi/pi*(1-phi/(k*A0)) + 2*zeta*A0
	return output
	
def b1(A0):
	if k*A0 <= phi:
		output = k*A0
	else:
		output = 1/pi*(k*A0*(pi/2-asin(1-2*phi/(k*A0))) \
		- 2*(1-2*phi/(k*A0))*sqrt(phi*(k*A0-phi)))
	return output

# Define the ode function
def du(t,u):
	udot = zeros(3)
	udot[0] = u[1]
	fd = u[2] + 2*zeta*u[1]
	udot[1] = -u[0] - eps*fd + eps*Gamma*sin(Omega*t)
	udot[2] = k*sign(1-u[2]/phi*sign(u[1]))*u[1]
	return udot

def du_asym(t,u):

	udot = zeros(2)
	
	x = u[0]
	y = u[1]
	
	A0 = sqrt(x**2+y**2)
	theta0 = atan2(y,x)
	
	dA0deta1 = -a1(A0)/2 - Gamma/2*sin(theta0)
	A0dtheta0deta1 = -alpha*A0 + b1(A0)/2 - Gamma/2*cos(theta0)
	
	dxdeta1 = dA0deta1*cos(theta0) - A0dtheta0deta1*sin(theta0)
	dydeta1 = dA0deta1*sin(theta0) + A0dtheta0deta1*cos(theta0)
	
	udot[0] = dxdeta1
	udot[1] = dydeta1
	
	return udot

def f_res(A0_dum):
	output = (a1(A0_dum)/Gamma)**2 \
				+ (2/Gamma)**2*(A0_dum*alpha-b1(A0_dum)/2)**2 - 1
	return output

# March in time and calculate numerical solution evolution using RK4 method
for n in range(0,N-1):

	######## Numerical Solution ########

	# Extract state at time step n
	t = t_state[n]
	x = x_state[n]
	xdot = xdot_state[n]
	z = z_state[n]
	
	# Allocate memory for vector for current state
	u = zeros(3)
	
	# Assign current state
	u[0] = x
	u[1] = xdot
	u[2] = z
	
	# Calculate increments according to RK4
	k1 = dt*du(t,u)
	k2 = dt*du(t+dt/2,u+k1/2)
	k3 = dt*du(t+dt/2,u+k2/2)
	k4 = dt*du(t+dt,u+k3)
	
	# Calcualte predicted value at n+1 timestep
	unp1 = u + 1/6*(k1+2*k2+2*k3+k4)
	
	# If the value of z (hysteretic internal variable) has gone outside
	# of the bounds [-phi,phi], map back to phi or -phi and recalculate
	# predictor/corrector solution.
	if unp1[2] > phi:

		u[2] = phi
		
		k1 = dt*du(t,u)
		k2 = dt*du(t+dt/2,u+k1/2)
		k3 = dt*du(t+dt/2,u+k2/2)
		k4 = dt*du(t+dt,u+k3)
		
		unp1 = u + 1/6*(k1+2*k2+2*k3+k4)
		
	elif unp1[2] < -phi:

		u[2] = -phi
		
		k1 = dt*du(t,u)
		k2 = dt*du(t+dt/2,u+k1/2)
		k3 = dt*du(t+dt/2,u+k2/2)
		k4 = dt*du(t+dt,u+k3)
		
		unp1 = u + 1/6*(k1+2*k2+2*k3+k4)
	
	# Assign output variables to history at n+1 timestep
	t_state[n+1] = t + dt
	x_state[n+1] = unp1[0]
	xdot_state[n+1] = unp1[1]
	z_state[n+1] = unp1[2]
	
	######## Asymptotic Solution ########
	
	eta_1 = eta1_state[n]
	xc = xc_state[n]
	yc = yc_state[n]
	A0 = A0_state[n]
	theta0 = theta0_state[n]
	
	# Allocate memory for vector for current state
	u = zeros(2)
	
	# Assign current state
	u[0] = xc
	u[1] = yc
	
	# Calculate increments according to RK4
	k1 = deta_1*du_asym(eta_1,u)
	k2 = deta_1*du_asym(eta_1+deta_1/2,u+k1/2)
	k3 = deta_1*du_asym(eta_1+deta_1/2,u+k2/2)
	k4 = deta_1*du_asym(eta_1+deta_1,u+k3)
	
	# Calcualte predicted value at n+1 timestep
	unp1 = u + 1/6*(k1+2*k2+2*k3+k4)
	
	xc_np1 = unp1[0]
	yc_np1 = unp1[1]
	A0_np1 = sqrt(xc_np1**2+yc_np1**2)
	theta0_np1 = atan2(yc_np1,xc_np1)
	xasym_np1 = A0_np1*sin(Omega*t+theta0_np1)
	
	# Assign output variables to history at n+1 timestep
	eta1_state[n+1] = eta_1+deta_1
	xc_state[n+1] = xc_np1
	yc_state[n+1] = yc_np1
	A0_state[n+1] = A0_np1
	theta0_state[n+1] = theta0_np1
	xasym_state[n+1] = xasym_np1
	
	######## Steady-State Linear Analytical Solution ########
	# Compare against this solution for verification and
	# to make sure the parameters are actually doing something
	# to the solution
	
	X0_SS = eps*Gamma/sqrt((1-Omega**2/(1)**2)**2+(2*eps*zeta*Omega/(1))**2)
	phi_SS = atan2(-2*eps*zeta*Omega/(1),1-Omega**2/(1)**2)
	
	x_analytical[n+1] = X0_SS*sin(Omega*t+phi_SS)


######## Calculate Fixed Point ########
A0_fixed = scipy.optimize.fsolve(f_res,0.1)
theta0_fixed = atan2(-a1(A0_fixed)/2,-(alpha*A0_fixed-b1(A0_fixed)/2))

x_fixed_plot = A0_fixed*cos(theta0_fixed)
y_fixed_plot = A0_fixed*sin(theta0_fixed)

######## Plot Solution ########

# Plot output states
fig, axs = plt.subplots(2,2)

# Displacement
axs[0,0].plot(t_state,x_state,t_state,xasym_state,t_state,x_analytical,'--')
axs[0,0].set_xlabel('Time')
axs[0,0].set_ylabel('x')

# Velocity
axs[0,1].plot(t_state,A0_state)
axs[0,1].set_xlabel('Time')
axs[0,1].set_ylabel('Asymptotic Amplitude')

# Internal Variable
axs[1,0].plot(t_state,theta0_state)
axs[1,0].set_xlabel('Time')
axs[1,0].set_ylabel('Asymptotic Phase')

# Hysteresis
axs[1,1].plot(x_state,z_state)
axs[1,1].set_xlabel('x')
axs[1,1].set_ylabel('z')

x_flow_plot = multiply(A0_state,cos(theta0_state))
y_flow_plot = multiply(A0_state,sin(theta0_state))

######## Calculate and Plot Vector Flow Field ########
x, y = mgrid[1:-1:30j, 1:-1:30j]#mgrid[3.4450:3.4275:50j, -3.6575:-3.6425:50j]

A0 = sqrt(x**2 + y**2)
theta0 = arctan2(y, x)

dA0 = zeros_like(x)
dtheta0 = zeros_like(x)

for i in range(0,len(dA0)-1):
	for j in range(0,len(dtheta0)-1):
		dA0[i,j] = -a1(A0[i,j])/2 - Gamma/2*sin(theta0[i,j])
		dtheta0[i,j] = 1/A0[i,j]*(-alpha*A0[i,j] + b1(A0[i,j])/2 \
											- Gamma/2*cos(theta0[i,j]))


dx = dA0*cos(theta0) - dtheta0*sin(theta0)
dy = dA0*sin(theta0) + dtheta0*cos(theta0)

norm_dr = sqrt(dx**2 + dy**2)

# Calculate linear limit
A0_lin = phi/k#Gamma/(2*sqrt((alpha-k/2)**2+zeta**2))
theta_lin = linspace(0,2*pi)
abscissa_lin = A0_lin*cos(theta_lin)
ordinate_lin = A0_lin*sin(theta_lin)

fig, ax = plt.subplots()
ax.quiver(x, y, dx/norm_dr, dy/norm_dr)
ax.plot(x_flow_plot,y_flow_plot,x_fixed_plot,y_fixed_plot,'ro')
ax.plot(abscissa_lin,ordinate_lin,'g--')
ax.set(aspect=1, title='Flow Field', xlabel=' ', ylabel=' ')
ax.set_xlabel(r'$A_{0}cos(\theta_{0})$')
ax.set_ylabel(r'$A_{0}sin(\theta_{0})$')
plt.show()

