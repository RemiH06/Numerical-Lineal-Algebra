import numpy as np

def algoritmo_qr(A, precision=1e-7, max_iter=1000):
    """
    Calcula todos los valores propios y vectores propios usando el algoritmo QR.
    Entradas:
      - A: matriz cuadrada de coeficientes.
      - precision: criterio de convergencia para elementos fuera de la diagonal.
      - max_iter: máximo número de iteraciones permitidas.
    Salidas:
      - eigenvalues: array con todos los valores propios.
      - eigenvectors: matriz con vectores propios en columnas.
      - k: número de iteraciones realizadas.
    Si el método no converge en 'max_iter', lanza un ValueError.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape

    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")

    # Inicializar la matriz de vectores propios como identidad
    V = np.eye(n)
    
    # Copiar A para trabajar con ella
    Ak = A.copy()

    for k in range(1, max_iter + 1):
        # Descomposición QR de Ak
        Q, R = np.linalg.qr(Ak)
        
        # Actualizar Ak = R * Q
        Ak = R @ Q
        
        # Acumular los vectores propios
        V = V @ Q
        
        # Verificar convergencia: elementos fuera de la diagonal deben ser pequeños
        # Extraer elementos fuera de la diagonal
        off_diagonal = Ak - np.diag(np.diag(Ak))
        max_off_diagonal = np.max(np.abs(off_diagonal))
        
        if max_off_diagonal < precision:
            # Extraer valores propios de la diagonal
            eigenvalues = np.diag(Ak)
            
            # Los vectores propios están en las columnas de V
            eigenvectors = V
            
            return eigenvalues, eigenvectors, k
    
    # Si llegó aquí, no convergió
    raise ValueError(f"Algoritmo QR no alcanzó la precisión deseada en {max_iter} iteraciones.")


def demo_algoritmo_qr():
    """
    Demostración del algoritmo QR para calcular todos los valores y vectores propios.
    """
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    print("\n" + "="*80)
    print("DEMOSTRACIÓN: ALGORITMO QR")
    print("="*80)

    # Matriz dada en el problema
    A = np.array([
        [8,  3,  1,   7],
        [3,  3,  5,   4],
        [1,  5,  4,  -2],
        [7,  4, -2,   2]
    ], dtype=float)

    print("\nMatriz A:")
    print(A)
    print(f"\nPrecisión utilizada: 0.0000001 (1e-7)")

    try:
        eigenvalues, eigenvectors, k = algoritmo_qr(A, precision=1e-7, max_iter=10000)
        
        print(f"\nAlgoritmo QR convergió en {k} iteraciones")
        print("\n" + "-"*80)
        print("VALORES PROPIOS:")
        print("-"*80)
        for i, lam in enumerate(eigenvalues):
            print(f"λ_{i+1} = {lam:.10f}")
        
        print("\n" + "-"*80)
        print("VECTORES PROPIOS (en columnas):")
        print("-"*80)
        print(eigenvectors)
        
        print("\n" + "-"*80)
        print("VECTORES PROPIOS INDIVIDUALES:")
        print("-"*80)
        for i in range(len(eigenvalues)):
            print(f"\nVector propio v_{i+1} (asociado a λ_{i+1} = {eigenvalues[i]:.10f}):")
            print(eigenvectors[:, i])

        # Verificación: A * v = λ * v
        print("\n" + "="*80)
        print("VERIFICACIÓN: A * v_i = λ_i * v_i")
        print("="*80)
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lam = eigenvalues[i]
            Av = A @ v
            lam_v = lam * v
            error = np.linalg.norm(Av - lam_v)
            print(f"\nVector propio {i+1}: ||A*v - λ*v|| = {error:.2e}")

    except ValueError as e:
        print(f"\nError: {e}")


def demo():
    demo_algoritmo_qr()

demo()