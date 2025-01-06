
import sympy as smp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sympy.printing import latex


t, g, m, omega = smp.symbols('t g m \omega')
x, y, z = smp.symbols(r'x, y, z', cls=smp.Function)
r, theta, f = smp.symbols(r'r, \theta, f', cls=smp.Function)
x = x(t)
y = y(t)
z = z(t)
r = r(t)
theta = theta(t)
f = f(r)


f = r**2
f_f = smp.lambdify(r, f)

x = r*smp.cos(theta)
y = r*smp.sin(theta)
z = f

x_d = smp.diff(x, t)
y_d = smp.diff(y, t)
z_d = smp.diff(z, t)

r_d = smp.diff(r, t)
r_dd = smp.diff(r_d, t)

v2 = smp.simplify(x_d**2 + y_d**2 + z_d**2)
T = 0.5 * m * v2
V = m * g * z
L = smp.simplify(T - V)

LE = smp.diff(L, r) - smp.diff(smp.diff(L, r_d), t).simplify()
sol = smp.solve([LE], [r_dd], simplify = True, rational=False)

sol[r_dd] = sol[r_dd].subs(smp.diff(theta, t), omega)


dvdt_f = smp.lambdify((t, g, m, omega, r, r_d), sol[r_dd])
drdt_f = smp.lambdify(r_d, r_d)

def dSdt(S, t, g, m, omega):
    r, v = S
    return [
        drdt_f(v),
        dvdt_f(t, g, m, omega, r, v),
    ]

t = np.linspace(0, 10, 1001)
r0, v0 = 10, 0
g = 9.81
m=1
omega = np.sqrt(g/5) +0.01
ans = odeint(dSdt, y0=[r0, v0], t=t, args=(g, m, omega))


R = ans.T[0]
R_d = ans.T[1]
th = t*omega
X = R*np.cos(th)
Y = R*np.sin(th)
Z = f_f(R)


def f_cart(x, y):
    return f_f(np.sqrt(x**2 + y**2))



fig = plt.figure()

ax = fig.add_subplot(3, 6, (1, 6))
ax.plot(t, R, c='red')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.set_xlabel('t')
ax.set_ylabel('r(t)')
ax.set_title('r(t) plot in time')

ax = fig.add_subplot(3, 6, (7, 8), projection = '3d')
ax.view_init(25, 35)
ax.grid(True)
x_ax, y_ax = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
z_plot = f_cart(x_ax, y_ax)
ax.plot_surface(x_ax, y_ax, z_plot, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none', zorder = 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'${latex(f)}$')

ax = fig.add_subplot(3, 6, (13, 14), projection = '3d')
ax.view_init(25, 35)
ax.grid(True)
ax.plot3D(X, Y, Z, color = 'red', zorder = -0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r(t)')
ax.set_title('r(t) in 3d')

ax = fig.add_subplot(3, 6, (9, 18))
ax.grid(True)
Rm, Vm = np.meshgrid(np.linspace(min(R), max(R), 100), np.linspace(min(R_d),max(R_d),100))
Rdot = drdt_f(Vm)
Vdot = dvdt_f(0, g, m, omega, Rm, Vm)
ax.streamplot(Rm, Vm, Rdot, Vdot)
ax.plot(R, R_d, c='red')
ax.scatter(r0, v0, c='red')
ax.set_xlabel('r')
ax.set_ylabel('v')
ax.set_title('phase plane')

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
