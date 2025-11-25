import numpy as np

def F_sistema(x):
    """
    Evalúa el sistema no lineal:
        f1(x) = 3 x1 - cos(x2 x3) - 1/2
        f2(x) = x1^2 - 81 (x2 + 0.1)^2 + sen(x3) + 1.06
        f3(x) = e^{-(x1 x2)} + 20 x3 + (10π - 3)/3

    Entrada:
      - x: vector (x1, x2, x3).

    Salida:
      - F(x): vector [f1, f2, f3].
    """
    x1, x2, x3 = x
    f1 = 3.0 * x1 - np.cos(x2 * x3) - 0.5
    f2 = x1**2 - 81.0 * (x2 + 0.1)**2 + np.sin(x3) + 1.06
    f3 = np.exp(-x1 * x2) + 20.0 * x3 + (10.0 * np.pi - 3.0) / 3.0
    return np.array([f1, f2, f3], dtype=float)


def J_sistema(x):
    """
    Jacobiano del sistema anterior.

        f1 = 3 x1 - cos(x2 x3) - 1/2
        f2 = x1^2 - 81 (x2 + 0.1)^2 + sen(x3) + 1.06
        f3 = e^{-(x1 x2)} + 20 x3 + (10π - 3)/3

    Se tiene:
        df1/dx1 = 3
        df1/dx2 =  x3 sen(x2 x3)
        df1/dx3 =  x2 sen(x2 x3)

        df2/dx1 = 2 x1
        df2/dx2 = -162 (x2 + 0.1)
        df2/dx3 =  cos(x3)

        df3/dx1 = -x2 e^{-(x1 x2)}
        df3/dx2 = -x1 e^{-(x1 x2)}
        df3/dx3 = 20

    Entrada:
      - x: vector (x1, x2, x3).

    Salida:
      - J(x): matriz 3x3 con las derivadas parciales.
    """
    x1, x2, x3 = x
    j11 = 3.0
    j12 = x3 * np.sin(x2 * x3)
    j13 = x2 * np.sin(x2 * x3)

    j21 = 2.0 * x1
    j22 = -162.0 * (x2 + 0.1)
    j23 = np.cos(x3)

    exp_term = np.exp(-x1 * x2)
    j31 = -x2 * exp_term
    j32 = -x1 * exp_term
    j33 = 20.0

    return np.array([[j11, j12, j13],
                     [j21, j22, j23],
                     [j31, j32, j33]], dtype=float)


def phi(F, x):
    """
    Función objetivo para el método de descenso más rápido:

        φ(x) = 1/2 * ||F(x)||_2^2

    Entrada:
      - F: función que recibe x y devuelve F(x).
      - x: vector punto de evaluación.

    Salida:
      - valor escalar φ(x).
    """
    Fx = np.array(F(x), dtype=float).flatten()
    return 0.5 * np.dot(Fx, Fx)


def grad_phi(F, J, x):
    """
    Gradiente de φ(x) = 1/2 * ||F(x)||_2^2.

    Se cumple:
        ∇φ(x) = J(x)^T F(x)

    Entradas:
      - F: función que recibe x y devuelve F(x).
      - J: jacobiano del sistema.
      - x: vector punto de evaluación.

    Salida:
      - gradiente ∇φ(x) como vector de dimensión n.
    """
    Fx = np.array(F(x), dtype=float).reshape(-1, 1)
    Jx = np.array(J(x), dtype=float)
    g = Jx.T @ Fx
    return g.flatten()


def descenso_mas_rapido(F, J, x0, num_iter=2):
    """
    Método del descenso más rápido (gradient descent) aplicado a φ(x) = 1/2 ||F(x)||^2.

    En cada iteración:
        d^(k) = -∇φ(x^(k))
        x^(k+1) = x^(k) + α_k d^(k)

    donde el paso α_k se selecciona mediante una búsqueda de línea (regla de Armijo).

    Entradas:
      - F: función que recibe x y devuelve F(x).
      - J: función que recibe x y devuelve el jacobiano J(x).
      - x0: vector inicial.
      - num_iter: número de iteraciones a realizar (para el problema se piden 2).

    Salidas:
      - x: vector obtenido tras 'num_iter' iteraciones (X^(num_iter)).
      - historial: lista con los vectores aproximados en cada iteración (incluye x0).
    """
    x = np.array(x0, dtype=float).flatten()
    if x.size == 0:
        raise ValueError("El vector inicial x0 no puede ser vacío.")

    historial = [x.copy()]

    for k in range(1, num_iter + 1):
        g = grad_phi(F, J, x)
        d = -g
        gTd = np.dot(g, d)

        # Búsqueda de línea tipo Armijo porque es la que encontré en internet
        alpha = 1.0
        c = 1e-4
        beta = 0.5
        phi_x = phi(F, x)

        while True:
            x_trial = x + alpha * d
            phi_trial = phi(F, x_trial)
            if phi_trial <= phi_x + c * alpha * gTd:
                break
            alpha *= beta
            if alpha < 1e-12:
                break

        x = x + alpha * d
        historial.append(x.copy())

    return x, historial


def demo_descenso():
    """
    Demostración del método de descenso más rápido para el sistema F(x) = 0.

    - Se define φ(x) = 1/2 ||F(x)||^2.
    - Se realizan exactamente 2 iteraciones a partir de x0 = (0,0,0).
    - Se imprime el vector en cada iteración, la norma infinito del gradiente
      y el paso α_k usado.

    Devuelve:
      - x2: vector X^(2) que pide el enunciado.
    """
    np.set_printoptions(precision=10, suppress=True, linewidth=140)

    print("\n" + "="*100)
    print("MÉTODO DE DESCENSO MÁS RÁPIDO (2 ITERACIONES)")
    print("="*100)

    x0 = np.array([0.0, 0.0, 0.0])
    num_iter = 2

    print(f"\nVector inicial x0 = {x0}")
    print(f"Número de iteraciones solicitadas = {num_iter}\n")

    print("Iteraciones del descenso más rápido:")
    print("  k              x1                  x2                  x3             ||grad_phi||_inf        alpha")
    print("----------------------------------------------------------------------------------------------------------------")

    x = x0.copy()
    g0 = grad_phi(F_sistema, J_sistema, x)
    norm_g0 = np.linalg.norm(g0, ord=np.inf)
    print(f"{0:3d}  {x[0]: .12f}  {x[1]: .12f}  {x[2]: .12f}  {norm_g0: .3e}          ---")

    for k in range(1, num_iter + 1):
        g = grad_phi(F_sistema, J_sistema, x)
        d = -g
        gTd = np.dot(g, d)
        norm_g = np.linalg.norm(g, ord=np.inf)

        # Búsqueda de línea (Armijo)
        alpha = 1.0
        c = 1e-4
        beta = 0.5
        phi_x = phi(F_sistema, x)

        while True:
            x_trial = x + alpha * d
            phi_trial = phi(F_sistema, x_trial)
            if phi_trial <= phi_x + c * alpha * gTd:
                break
            alpha *= beta
            if alpha < 1e-12:
                break

        x = x + alpha * d

        print(f"{k:3d}  {x[0]: .12f}  {x[1]: .12f}  {x[2]: .12f}  {norm_g: .3e}      {alpha: .3e}")

    print("\nVector obtenido tras 2 iteraciones (X^(2)):")
    print(x)

    return x


def demo():
    """
    Ejecuta la demostración del método de descenso más rápido
    para el sistema dado y muestra el vector X^(2).
    """
    X2 = demo_descenso()
    return X2


# Llamar a la demo principal
demo()
