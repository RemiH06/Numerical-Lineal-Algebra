import numpy as np

def metodo_potencia(A, tolerancia=1e-7, max_iter=1000, x0=None):
    """
    Calcula el valor propio dominante y su vector propio usando el método de la potencia.
    Entradas:
      - A: matriz cuadrada de coeficientes.
      - tolerancia: criterio de paro |μ^(m) - μ^(m-1)| < tolerancia.
      - max_iter: máximo número de iteraciones permitidas.
      - x0: vector inicial (opcional). Si no se da, se usa un vector aleatorio.
    Salidas:
      - eigenvalue: valor propio dominante (μ).
      - eigenvector: vector propio asociado (normalizado).
      - k: número de iteraciones realizadas.
    Si el método no converge en 'max_iter', lanza un ValueError con un mensaje claro.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape

    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")

    # Inicialización: vector inicial aleatorio o proporcionado
    if x0 is None:
        np.random.seed(42)  # Para reproducibilidad
        x = np.random.rand(n)
    else:
        x = np.array(x0, dtype=float).flatten()

    # Normalizar el vector inicial
    x = x / np.linalg.norm(x)

    mu_prev = 0.0

    for k in range(1, max_iter + 1):
        # Multiplicar A por el vector actual
        y = A @ x

        # Calcular el valor propio aproximado (cociente de Rayleigh)
        mu = np.dot(x, y)

        # Normalizar el nuevo vector
        x = y / np.linalg.norm(y)

        # Criterio de finalización: |μ^(m) - μ^(m-1)| < tolerancia
        if abs(mu - mu_prev) < tolerancia:
            return mu, x, k

        mu_prev = mu

    # Si llegó aquí, no cumplió el criterio dentro de max_iter
    raise ValueError(f"Método de la potencia no alcanzó la tolerancia en {max_iter} iteraciones.")


def demo_metodo_potencia():
    """
    Demostración del método de la potencia para calcular el valor propio dominante.
    Compara el resultado con la función eig de numpy.linalg.
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=100)

    print("\n" + "="*80)
    print("DEMOSTRACIÓN: MÉTODO DE LA POTENCIA")
    print("="*80)

    # --- Caso 1: Matriz 8x8 con valor propio dominante real ---
    # Creamos una matriz simétrica para garantizar valores propios reales
    np.random.seed(123)
    M = np.random.rand(8, 8)
    A1 = M + M.T  # Hacer simétrica
    # Ajustar para tener un valor propio dominante más claro
    A1 = A1 + 10 * np.eye(8)

    print("\nCASO 1: Matriz 8x8 simétrica con valor propio dominante claro")
    print("Matriz A:")
    print(A1)

    try:
        eigenvalue, eigenvector, k = metodo_potencia(A1, tolerancia=1e-7, max_iter=1000)
        print(f"\nMétodo de la Potencia (en {k} iteraciones):")
        print(f"Valor propio dominante: {eigenvalue:.10f}")
        print(f"Vector propio asociado:")
        print(eigenvector)

        # Comparación con numpy.linalg.eig
        eigenvalues_np, eigenvectors_np = np.linalg.eig(A1)
        # Ordenar por valor absoluto (el dominante es el de mayor valor absoluto)
        idx = np.argmax(np.abs(eigenvalues_np))
        eigenvalue_np = eigenvalues_np[idx]
        eigenvector_np = eigenvectors_np[:, idx]

        # Normalizar el vector propio de numpy para comparación
        eigenvector_np = eigenvector_np / np.linalg.norm(eigenvector_np)

        # Ajustar signo si es necesario (los vectores propios pueden diferir en signo)
        if np.dot(eigenvector, eigenvector_np) < 0:
            eigenvector_np = -eigenvector_np

        print(f"\nnumpy.linalg.eig:")
        print(f"Valor propio dominante: {eigenvalue_np.real:.10f}")
        print(f"Vector propio asociado:")
        print(eigenvector_np.real)

        print(f"\nDiferencia en valor propio: {abs(eigenvalue - eigenvalue_np.real):.2e}")
        print(f"Diferencia en vector propio (norma): {np.linalg.norm(eigenvector - eigenvector_np.real):.2e}")

    except ValueError as e:
        print(f"\nMétodo de la Potencia: {e}")

    # --- Caso 2: Matriz tridiagonal 8x8 ---
    print("\n" + "="*80)
    print("\nCASO 2: Matriz tridiagonal 8x8")
    A2 = np.array([
        [2, 1, 0, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0, 0],
        [0, 1, 2, 1, 0, 0, 0, 0],
        [0, 0, 1, 2, 1, 0, 0, 0],
        [0, 0, 0, 1, 2, 1, 0, 0],
        [0, 0, 0, 0, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 0, 1, 2]
    ], dtype=float)

    print("Matriz A:")
    print(A2)

    try:
        eigenvalue, eigenvector, k = metodo_potencia(A2, tolerancia=1e-7, max_iter=1000)
        print(f"\nMétodo de la Potencia (en {k} iteraciones):")
        print(f"Valor propio dominante: {eigenvalue:.10f}")
        print(f"Vector propio asociado:")
        print(eigenvector)

        # Comparación con numpy
        eigenvalues_np, eigenvectors_np = np.linalg.eig(A2)
        idx = np.argmax(np.abs(eigenvalues_np))
        eigenvalue_np = eigenvalues_np[idx]
        eigenvector_np = eigenvectors_np[:, idx]
        eigenvector_np = eigenvector_np / np.linalg.norm(eigenvector_np)

        if np.dot(eigenvector, eigenvector_np) < 0:
            eigenvector_np = -eigenvector_np
        
        print(f"\nnumpy.linalg.eig:")
        print(f"Valor propio dominante: {eigenvalue_np.real:.10f}")
        print(f"Vector propio asociado:")
        print(eigenvector_np.real)

        print(f"\nDiferencia en valor propio: {abs(eigenvalue - eigenvalue_np.real):.2e}")
        print(f"Diferencia en vector propio (norma): {np.linalg.norm(eigenvector - eigenvector_np.real):.2e}")

    except ValueError as e:
        print(f"\nMétodo de la Potencia: {e}")

    # --- Caso 3: Matriz que podría no converger fácilmente ---
    print("\n" + "="*80)
    print("\nCASO 3: Matriz con valores propios cercanos (convergencia lenta)")
    A3 = np.array([
        [5.1, 1, 0, 0, 0, 0, 0, 0],
        [1, 5.0, 1, 0, 0, 0, 0, 0],
        [0, 1, 4.9, 1, 0, 0, 0, 0],
        [0, 0, 1, 4.8, 1, 0, 0, 0],
        [0, 0, 0, 1, 4.7, 1, 0, 0],
        [0, 0, 0, 0, 1, 4.6, 1, 0],
        [0, 0, 0, 0, 0, 1, 4.5, 1],
        [0, 0, 0, 0, 0, 0, 1, 4.4]
    ], dtype=float)

    print("Matriz A:")
    print(A3)

    try:
        eigenvalue, eigenvector, k = metodo_potencia(A3, tolerancia=1e-7, max_iter=2000)
        print(f"\nMétodo de la Potencia (en {k} iteraciones):")
        print(f"Valor propio dominante: {eigenvalue:.10f}")

        # Comparación con numpy
        eigenvalues_np = np.linalg.eig(A3)[0]
        idx = np.argmax(np.abs(eigenvalues_np))
        eigenvalue_np = eigenvalues_np[idx]
        
        print(f"\nnumpy.linalg.eig:")
        print(f"Valor propio dominante: {eigenvalue_np.real:.10f}")
        print(f"Diferencia: {abs(eigenvalue - eigenvalue_np.real):.2e}")

    except ValueError as e:
        print(f"\nMétodo de la Potencia: {e}")


def demo():
    demo_metodo_potencia()

demo()