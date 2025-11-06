import numpy as np

def punto_fijo(g, x0, tolerancia=1e-8, max_iter=1000):
    """
    Resuelve x = g(x) con el método de iteración de punto fijo.
    Entradas:
      - g: función que recibe un vector x (numpy.array) y devuelve g(x).
      - x0: vector inicial.
      - tolerancia: criterio de paro relativo en norma infinito.
      - max_iter: máximo número de iteraciones permitidas.
    Salidas:
      - x: aproximación al punto fijo.
      - k: número de iteraciones realizadas.
    Si el método no converge en 'max_iter', lanza un ValueError con un mensaje claro.
    """
    x = np.array(x0, dtype=float).flatten()
    if x.size == 0:
        raise ValueError("El vector inicial x0 no puede ser vacío.")
    
    # Verificación mínima de que g(x0) es compatible
    try:
        gx = np.array(g(x), dtype=float).flatten()
    except Exception as e:
        raise ValueError(f"Error al evaluar g(x0) en el punto inicial: {e}")
    
    if gx.size != x.size:
        raise ValueError("La función g(x) debe devolver un vector de la misma dimensión que x0.")
    
    for k in range(1, max_iter + 1):
        x_nuevo = np.array(g(x), dtype=float).flatten()
        if x_nuevo.size != x.size:
            raise ValueError("La dimensión de g(x) cambió durante las iteraciones.")
        
        # Criterio de finalización: ||x_{k} - x_{k-1}||_inf / ||x_k||_inf < tolerancia
        num = np.linalg.norm(x_nuevo - x, ord=np.inf)
        den = np.linalg.norm(x_nuevo, ord=np.inf)
        rel = num / den if den > 0 else num
        
        if rel < tolerancia:
            return x_nuevo, k
        
        x = x_nuevo
    
    raise ValueError(f"El método de punto fijo no alcanzó la tolerancia en {max_iter} iteraciones.")

def sistema_no_lineal(x):
    """
    Evalúa el sistema de ecuaciones no lineales:
        3x1 - cos(x2 x3) - 1/2 = 0
        x1^2 - 81(x2 + 0.1)^2 + sin x3 + 1.06 = 0
        e^{-(x1 x2)} + 20 x3 + (10π - 3)/3 = 0
    Entrada:
      - x: vector (x1, x2, x3).
    Salida:
      - f(x): vector con las tres ecuaciones evaluadas.
    """
    x1, x2, x3 = x
    f1 = 3.0 * x1 - np.cos(x2 * x3) - 0.5
    f2 = x1**2 - 81.0 * (x2 + 0.1)**2 + np.sin(x3) + 1.06
    f3 = np.exp(-x1 * x2) + 20.0 * x3 + (10.0 * np.pi - 3.0) / 3.0
    return np.array([f1, f2, f3], dtype=float)

def g_punto_fijo(x):
    """
    Función de iteración g(x) asociada al sistema anterior.
    Se despeja:
        x1 = (cos(x2 x3) + 1/2) / 3
        x2 = -0.1 + sqrt( (x1^2 + sin x3 + 1.06) / 81 )
        x3 = -( e^{-(x1 x2)} + (10π - 3)/3 ) / 20
    Entrada:
      - x: vector (x1, x2, x3).
    Salida:
      - g(x): nuevo vector.
    Lanza ValueError si la expresión bajo la raíz es negativa.
    """
    x1, x2, x3 = x

    g1 = (np.cos(x2 * x3) + 0.5) / 3.0

    aux = (x1**2 + np.sin(x3) + 1.06) / 81.0
    if aux < 0:
        raise ValueError("La expresión bajo la raíz cuadrada es negativa; "
                         "esta elección de g(x) no es válida para el x actual.")
    g2 = -0.1 + np.sqrt(aux)

    g3 = -(np.exp(-x1 * x2) + (10.0 * np.pi - 3.0) / 3.0) / 20.0

    return np.array([g1, g2, g3], dtype=float)

def demo_punto_fijo():
    """
    Demostración del método de punto fijo para el sistema no lineal dado.
    - Se toma un único vector inicial x0 en D = [-1, 1]^3.
    - Se imprime CADA iteración hasta que el error relativo sea <= 1e-8.
    """
    np.set_printoptions(precision=10, suppress=True, linewidth=120)

    print("\n" + "="*80)
    print("MÉTODO DEL PUNTO FIJO PARA UN SISTEMA NO LINEAL EN R^3")
    print("="*80)

    print("\nSistema de ecuaciones:")
    print("  1)  3 x1 - cos(x2 x3) - 1/2 = 0")
    print("  2)  x1^2 - 81 (x2 + 0.1)^2 + sin(x3) + 1.06 = 0")
    print("  3)  e^{-(x1 x2)} + 20 x3 + (10π - 3)/3 = 0")

    print("\nDominio de trabajo (de la tarea):")
    print("  D = { (x1, x2, x3) ∈ R^3 : -1 ≤ xi ≤ 1, ∀i }")

    # Un solo vector inicial dentro de D (puedes cambiarlo si en tu tarea usaste otro)
    x0 = np.array([0.0, 0.0, 0.0])
    tolerancia = 1e-8
    max_iter = 1000

    print(f"\nVector inicial x0 = {x0}")
    print(f"Tolerancia pedida = {tolerancia:.0e}\n")

    print("Iteraciones del método de punto fijo:")
    print("  k        x1                x2                x3              error_relativo")
    print("--------------------------------------------------------------------------------")

    x = x0.copy()
    # Mostramos k = 0 (punto inicial, sin error aún)
    print(f"{0:3d}  {x[0]: .12f}  {x[1]: .12f}  {x[2]: .12f}        ---")

    for k in range(1, max_iter + 1):
        x_nuevo = g_punto_fijo(x)

        num = np.linalg.norm(x_nuevo - x, ord=np.inf)
        den = np.linalg.norm(x_nuevo, ord=np.inf)
        rel = num / den if den > 0 else num

        print(f"{k:3d}  {x_nuevo[0]: .12f}  {x_nuevo[1]: .12f}  {x_nuevo[2]: .12f}  {rel: .3e}")

        if rel < tolerancia:
            print("\nCriterio de paro alcanzado: error relativo < tolerancia.")
            print(f"Número total de iteraciones: {k}")
            print("\nSolución aproximada x*:")
            print(x_nuevo)

            fx = sistema_no_lineal(x_nuevo)
            norma_fx = np.linalg.norm(fx, ord=np.inf)
            print("\nValor de f(x*) (para verificar):")
            print(fx)
            print(f"\n||f(x*)||_inf = {norma_fx:.3e}")
            return

        x = x_nuevo

    # Si se sale del bucle sin haber parado
    raise ValueError(f"El método de punto fijo no alcanzó la tolerancia en {max_iter} iteraciones.")

def demo():
    demo_punto_fijo()

demo()
