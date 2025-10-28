import numpy as np

def biseccion(f, a, b, tolerancia=1e-8, max_iter=1000):
    """
    Resuelve f(x) = 0 usando el método de bisección.
    Entradas:
      - f: función a resolver.
      - a, b: extremos del intervalo [a, b].
      - tolerancia: criterio de paro relativo.
      - max_iter: número máximo de iteraciones.
    Salidas:
      - x: aproximación a la raíz de la función.
      - k: número de iteraciones realizadas.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("La función no cambia de signo en el intervalo proporcionado.")
    
    print(f"Intervalo inicial: [{a}, {b}]")
    
    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        print(f"Iteración {k}: c = {c}, f(c) = {f(c)}")
        if abs(f(c)) < tolerancia:
            print(f"Raíz encontrada: {c}")
            return c, k
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    raise ValueError(f"La bisección no convergió en {max_iter} iteraciones.")


def punto_fijo(g, x0, tolerancia=1e-8, max_iter=1000):
    """
    Resuelve f(x) = 0 usando el método de punto fijo.
    Entradas:
      - g: función de iteración que define el punto fijo.
      - x0: aproximación inicial.
      - tolerancia: criterio de paro relativo.
      - max_iter: número máximo de iteraciones.
    Salidas:
      - x: aproximación al punto fijo.
      - k: número de iteraciones realizadas.
    """
    print(f"Valor inicial: {x0}")
    
    for k in range(1, max_iter + 1):
        x = g(x0)
        print(f"Iteración {k}: x = {x}, g(x) = {g(x)}")
        if abs(x - x0) < tolerancia:
            print(f"Raíz encontrada: {x}")
            return x, k
        x0 = x
    
    raise ValueError(f"El método de punto fijo no convergió en {max_iter} iteraciones.")


def newton(f, df, x0, tolerancia=1e-8, max_iter=1000):
    """
    Resuelve f(x) = 0 usando el método de Newton.
    Entradas:
      - f: función a resolver.
      - df: derivada de la función f.
      - x0: aproximación inicial.
      - tolerancia: criterio de paro relativo.
      - max_iter: número máximo de iteraciones.
    Salidas:
      - x: aproximación a la raíz de la función.
      - k: número de iteraciones realizadas.
    """
    print(f"Valor inicial: {x0}")
    
    for k in range(1, max_iter + 1):
        fx0 = f(x0)
        dfx0 = df(x0)
        print(f"Iteración {k}: x0 = {x0}, f(x0) = {fx0}, f'(x0) = {dfx0}")
        if abs(fx0) < tolerancia:
            print(f"Raíz encontrada: {x0}")
            return x0, k
        if dfx0 == 0:
            raise ValueError(f"La derivada es cero en x = {x0}. El método no puede continuar.")
        
        x0 = x0 - fx0 / dfx0
    
    raise ValueError(f"El método de Newton no convergió en {max_iter} iteraciones.")


def demo():
    """
    Demostración de los tres métodos con diferentes ecuaciones.
    """

    print("="*80)
    print("Métodos de Bisección, Punto Fijo y Newton")
    print("="*80)
    
    # --- Caso 1: Método de Bisección ---
    print("\nMétodo de Bisección:")
    f_biseccion = lambda x: np.cos(x) - x
    a, b = 0, 1
    try:
        raiz_biseccion, k_biseccion = biseccion(f_biseccion, a, b)
        print(f"Raíz encontrada: {raiz_biseccion}")
        print(f"Iteraciones: {k_biseccion}")
    except ValueError as e:
        print(f"Error: {e}")

    # --- Caso 2: Método de Punto Fijo ---
    print("\nMétodo de Punto Fijo:")
    g_punto_fijo = lambda x: np.cos(x)
    x0_punto_fijo = 0.5
    try:
        raiz_punto_fijo, k_punto_fijo = punto_fijo(g_punto_fijo, x0_punto_fijo)
        print(f"Raíz encontrada: {raiz_punto_fijo}")
        print(f"Iteraciones: {k_punto_fijo}")
    except ValueError as e:
        print(f"Error: {e}")

    # --- Caso 3: Método de Newton ---
    print("\nMétodo de Newton:")
    f_newton = lambda x: x**2 - 2
    df_newton = lambda x: 2*x
    x0_newton = 1.0
    try:
        raiz_newton, k_newton = newton(f_newton, df_newton, x0_newton)
        print(f"Raíz encontrada: {raiz_newton}")
        print(f"Iteraciones: {k_newton}")
    except ValueError as e:
        print(f"Error: {e}")


demo()