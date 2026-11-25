import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# ---------- Constants i dades de l'enunciat ----------
cv    = 3698      # calor especific      [J/(kg K)]
rho   = 1081      # densitat              [kg/m^3]
kappa = 0.56      # conductivitat termica [W/(m K)]
sigma = 0.472     # conductivitat electrica [S/m]

# ---------- Geometria y excitacio ----------
d      = 0.02                 # distancia entre electrodes = gruixut de la paret [m]
V_pic  = 40.0                 # potencial de pic [V]
V_ef   = V_pic / np.sqrt(2)   # potencial efectiu [V]
Tc     = 36.5                 # temperatura corporal [C]

# ---------- Adimensionalitzacio ----------
alpha = kappa / (cv * rho)          # difusivitat termica [m^2/s]
x0    = d                           # escala de longitud [m]
t0    = d**2 / alpha                # escala de temps [s]
T0    = sigma * V_ef**2 / kappa     # escala de temperatura [K]

beta  = Tc / T0                     # condicio inicial y de contorn adimensional

print(f"alpha = {alpha:.4e} m^2/s")
print(f"x0 = {x0} m,  t0 = {t0:.3f} s,  T0 = {T0:.3f} K")
print(f"beta = Tc/T0 = {beta:.5f}")

# ---------- Adimensionalitzacio de llindars clinics ----------
T_muerte = 50.0 / T0     # punt des d'on hi ha mort cel·lular
T_max    = 80.0 / T0     # punt on comença a formar-se trombosi

# ---------- Discretitzacio ----------
N  = 101                 # nodes espacials
ta = 0.025               # temps final adimensional
dx = 1.0 / (N - 1)       # pas espacial (mida adimensional de la paret = 1)

print(f"dx = {dx}, ta = {ta}  ->  t fisico = {ta*t0:.1f} s")

def T_analitica(x, t, beta, n_terminos=500):
    # Suma de la serie del anexo, sólo con los términos impares n = 1, 3, 5, ...
    n    = np.arange(1, 2 * n_terminos, 2)                       # n impares
    bn   = 4.0 / (n * np.pi)                                     # coeficientes b_n de q = 1
    coef = bn * (1 - np.exp(-n**2 * np.pi**2 * t)) / (n**2 * np.pi**2)
    # matriz (N x n_terminos) de senos, multiplicada por los coeficientes
    return beta + np.sin(np.outer(x, n * np.pi)) @ coef

# Comprobación rápida: en t = 0 debe dar beta en todos los puntos
x_test = np.linspace(0, 1, N)
print("max|T(x,0) - beta| =", np.max(np.abs(T_analitica(x_test, 0.0, beta) - beta)))

def gauss_seidel(T_old, T, r, dt, tol, max_iter, hist):
    """
    Resuelve UN paso temporal de Euler implícito.

    T_old : temperatura en el instante n (fija; forma parte del término independiente b)
    T     : entra con el "guess" inicial y sale con la solución en n+1 (se modifica in situ)
    hist  : vector donde se guarda max|T^(k+1)-T^(k)| de cada iteración (para analizar convergencia)

    Devuelve el número de iteraciones usadas, o -1 si no convergió en max_iter.
    """
    N = T.shape[0]
    for k in range(max_iter):
        max_dif = 0.0
        for i in range(1, N - 1):              # sólo nodos interiores; T[0] y T[N-1] son el contorno
            nuevo = (T_old[i] + dt + r * (T[i - 1] + T[i + 1])) / (1.0 + 2.0 * r)
            dif = abs(nuevo - T[i])
            if dif > max_dif:
                max_dif = dif
            T[i] = nuevo                       # se sobrescribe YA: el siguiente nodo usará este valor nuevo
        hist[k] = max_dif
        if max_dif < tol:                      # criterio de parada: el barrido ya casi no cambia nada
            return k + 1
    return -1

def euler_implicito(N, r, ta, beta, tol=1e-12, max_iter=100000):
    """
    Integra desde t = 0 hasta t = ta con paso dt = r * dx^2.
    Devuelve x, T(x, ta) y el número de iteraciones de GS que se usaron en cada paso temporal.
    """
    dx = 1.0 / (N - 1)
    dt = r * dx**2
    n_pasos = int(round(ta / dt))
    assert abs(n_pasos * dt - ta) < 1e-12, "ta / dt no es entero: no se llega exactamente a ta"

    x = np.linspace(0.0, 1.0, N)
    T = np.full(N, beta)                       # condición inicial Y de contorno: todo a beta
    hist = np.zeros(max_iter)
    iters = np.zeros(n_pasos, dtype=int)

    for n in range(n_pasos):                   # bucle temporal
        T_old = T.copy()                       # T^n, congelado durante todo el paso
        # T entra como guess (= T^n, buen punto de partida porque cambia poco entre pasos)
        k = gauss_seidel(T_old, T, r, dt, tol, max_iter, hist)
        if k < 0:
            raise RuntimeError(f"Gauss-Seidel no convergió en el paso {n}")
        iters[n] = k
    return x, T, iters

# ---------- Paso 1: elegir parámetros ----------
# dt = r * dx^2, así que dar r equivale a dar dt. Número de pasos: 250 para r = 1, 500 para r = 0.5.
N   = 101
tol = 1e-12                    # claramente por debajo del error de truncamiento (ver sección 3b)
casos = {r"$\Delta t=\Delta x^2$": 1.0,
         r"$\Delta t=0.5\,\Delta x^2$": 0.5}

# ---------- Integracio en el temps y calcul de l'error numeric ----------

resultados = {}
for etiqueta, r in casos.items():
    x, T_num, iters = euler_implicito(N, r, ta, beta, tol)
    T_an  = T_analitica(x, ta, beta)           # solucio exacta al mateix instant ta
    error = np.abs(T_num - T_an)               # error punt a punt
    resultados[etiqueta] = (T_num, T_an, error, iters)
    print(f"r = {r:<4} -> pasos = {len(iters):>3},  iter. GS/paso (media) = {iters.mean():.1f},  "
          f"error máx = {error.max():.3e}")


#Crank-Nicolson

# Importem les llibreries necessaries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve


# Definim els parametres del problema


N = 101
xa = 1.0
ta = 0.025

dx = xa / (N - 1)
x = np.linspace(0, xa, N)

# Temperatura inicial i escala de temperatura
Tc = 36.5 + 273.15       # K
T0 = 674.286             # K
beta = Tc / T0

# Nombre de termes de la serie de Fourier
n_terms = 1000


# Definim la nostra solucio analitica

def sol_analitica(x, t):
    beta = (36.5 + 273.15) / 674.286

    T = np.full_like(x, beta, dtype=float)

    for n in range(1, 2000, 2):

        bn = 4 / (n * np.pi)

        T += bn * ((1 - np.exp(-n**2 * np.pi**2 * t))
                   / (n**2 * np.pi**2)) * np.sin(n * np.pi * x)

    return T


# Apliquem el mètode de Crank-Nicolson


def crank_nicolson(dt):

    # Nombre de passos temporals
    Nt = round(ta / dt)

    # Ajustem dt per arribar exactament a ta
    dt = ta / Nt

    r = dt / dx**2

    # Condicio inicial: increment de temperatura (respecte de la temperatura corporal)
    theta = np.zeros(N)

    # Contstruim la matriu A: diagonal principal 1+r i diagonals secundaries -r/2
    A = np.diag((1 + r) * np.ones(N-2))
    A += np.diag((-r/2) * np.ones(N-3), 1)
    A += np.diag((-r/2) * np.ones(N-3), -1)

    # Contruim la matriu B: diagonal principal 1-r i diagonals secundaries r/2
    B = np.diag((1 - r) * np.ones(N-2))
    B += np.diag((r/2) * np.ones(N-3), 1)
    B += np.diag((r/2) * np.ones(N-3), -1)

    # Factoritzacio LU de A (es calcula una sola vegada)
    LU = lu_factor(A)

    # Evolucio temporal
    for k in range(Nt):

        # Definim el terme independent
        b = B @ theta[1:-1] + dt * np.ones(N-2)

        # Resolem el sistema lineal
        theta[1:-1] = lu_solve(LU, b)

    # Recuperem la temperatura normalitzada absoluta
    T = theta + beta

    return T, theta, Nt, dt



# Passem a la resolucio dels dos casos amb els respectius dt


dt1 = dx**2
dt2 = 0.5 * dx**2

T1, theta1, Nt1, dt1 = crank_nicolson(dt1)
T2, theta2, Nt2, dt2 = crank_nicolson(dt2)

# Solucio analitica al temps final
T_exacta = sol_analitica(x, ta)


# Calculem els diferents errors de cada cas


error1 = np.abs(T1 - T_exacta)
error2 = np.abs(T2 - T_exacta)

# Error maxim
error_max1 = np.max(error1)
error_max2 = np.max(error2)

# Error quadratic mitja
error_mitja1 = np.sqrt(np.mean(error1**2))
error_mitja2 = np.sqrt(np.mean(error2**2))

# Mostrem els errors de cada cas
print("CAS 1: dt = dx^2")
print("Nombre de passos:", Nt1)
print("dt =", dt1)
print("Error maxim =", error_max1)
print("Error quadratic mitja =", error_mitja1)

print("\nCAS 2: dt = 0.5*dx^2")
print("Nombre de passos:", Nt2)
print("dt =", dt2)
print("Error maxim =", error_max2)
print("Error quadratic mitja =", error_mitja2)





# Expressem la temperaura en graus celcius


T_celsius = Tc + T0 * theta2 - 273.15



import numpy as np
import matplotlib.pyplot as plt
import math

# Definim els paràmetres del problema

alpha = 0.56/(3698*1081)

Nx = 101

# Definim els delta x i delta t
dx = 1/(Nx-1)
dt = [0.51 * dx**2, 0.49 * dx**2, 0.25 * dx**2]

# Definim els paràmetres normalitzats
x0 = 0.02
t0 = 0.02**2/alpha
tf = 0.025

# Coordenades de la zona malalta 
x1 = 0.0075
x2 = 0.0125

# Coordenades de la zona malalta normalitzades
x1n = x1/x0
x2n = x2/x0

# Definim les temperatures normalitzades
T0 = (0.472*(40/np.sqrt(2))**2)/0.56
Tc = (36.50 + 273.15)/T0
T50 = (50 + 273.15) / T0
T80 = (80 + 273.15) / T0

# Definim els punts espaials
x = np.linspace(0, 1, Nx)

# Definim la regió malalta
regio_malalta = (x >= 0.375) & (x <= 0.625)

# Definim la regió sana
regio_sana = (x < 0.375) | (x > 0.625)


# Definim la solució analítica

def T_analitica(x, t, Tc):
    T = Tc * np.ones(len(x))

    for n in range(1, 101):
        bn = 4 / ((2*n-1) * np.pi)
        
        T += bn * ((1 - np.exp(-(2*n-1)**2 * np.pi**2 * t))/ ((2*n-1)**2 * np.pi**2)) * np.sin((2*n-1) * np.pi * x)
        
    return T


# Llistes per guardar els resultats

T_f = []
errors = []

# Resolem numèricament l'equació

# Introduïm l'expressió de l'algorisme trobada pel mètode d'Euler explícit.

for s in dt:
    gamma = s/(dx)**2
    Nt = round(tf/s, None)         # Nombre de passes temporals

    # Condició inicial: T(x, 0) = Tc a tot el domini
    T = Tc * np.ones(Nx)

    # Bucle temporal 
    for i in range(Nt):

        T_nova = np.array(T)

        for j in range(1, Nx-1):
            T_nova[j] = T[j] + gamma * (T[j+1] - 2*T[j] + T[j-1]) + s
            
        T = np.array(T_nova)

    # Guardem el resultat final

    T_f.append(T)

    # Calculem la solució analítica al mateix temps
    T_exacta = T_analitica(x, Nt*s, Tc)

    # Calculem l'error
    error = np.abs(T - T_exacta)

    errors.append(error)


# Utilitzem el pas temporal estable a = 0.49 (amb dt[1]). També es pot fer servir a = 0.25 (amb dt[2])
s = dt[1]
gamma = s / dx**2

# Definim la temperatura inicial
T = Tc * np.ones(Nx)

# Temps inicial
temps = 0

# Simulació fins que s'arribi a una condició de seguretat
while True:

    # Temperatura màxima de la regió sana
    Tmax_sana = np.max(T[regio_sana])

    # Temperatura màxima de tot el teixit
    T_max = np.max(T)

    # Comprovem les condicions de seguretat
    if Tmax_sana >= T50 or T_max >= T80:
        break

    # Actualitzem la temperatura
    T_nova = T.copy()

    for j in range(1, Nx-1):
        T_nova[j] = T[j] + gamma * (T[j+1] - 2*T[j] + T[j-1]) + s

    T = T_nova

    # Actualitzem el temps
    temps += s

# Convertim el temps normalitzat a segons
temps_segons = temps * t0

print("Temps del tractament t =", temps_segons, "s")





# Gràfica conjunta dels errors dels tres mètodes

plt.figure(figsize=(10, 5))

# Euler explícit
plt.plot(x*0.02*100, errors[1], label='Euler explícit: Δt = 0.49 Δx²')
plt.plot(x*0.02*100, errors[2], label='Euler explícit: Δt = 0.25 Δx²')

# Euler implícit
for etiqueta, (T_num, T_an, error, iters) in resultados.items():
    plt.plot(x*0.02*100, error, label=f'Euler implícit: {etiqueta}')

# Crank-Nicolson
plt.plot(x*0.02*100, error1, label='Crank-Nicolson: Δt = Δx²')
plt.plot(x*0.02*100, error2, label='Crank-Nicolson: Δt = 0.5 Δx²')

plt.xlabel('x (cm)')
plt.ylabel('Error absolut (adimensional)')
plt.title('Comparació dels errors dels tres mètodes numèrics')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.grid('True')
plt.tight_layout()
plt.show()