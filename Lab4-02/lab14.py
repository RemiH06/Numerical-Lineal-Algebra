import numpy as np

def potencia_inversa(A, tolerancia=1e-7, max_iter=1000):
    """
    Calcula todos los valores propios y vectores propios usando el método de la potencia inversa.
    Entradas:
      - A: matriz cuadrada de coeficientes.
      - tolerancia: criterio de paro |λ^(m) - λ^(m-1)| < tolerancia.
      - max_iter: máximo número de iteraciones permitidas por cada valor propio.
    Salidas:
      - eigenvalues: lista con todos los valores propios.
      - eigenvectors: lista con todos los vectores propios (normalizados).
      - iterations: lista con el número de iteraciones para cada valor propio.
    Si el método no converge para algún valor propio, lanza un ValueError.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape

    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")

    eigenvalues = []
    eigenvectors = []
    iterations = []

    # Matriz para deflación
    A_deflated = A.copy()

    for idx in range(n):
        # Vector inicial aleatorio
        np.random.seed(42 + idx)
        x = np.random.rand(n)
        x = x / np.linalg.norm(x)

        lambda_prev = 0.0

        for k in range(1, max_iter + 1):
            try:
                # Resolver el sistema A_deflated * y = x
                y = np.linalg.solve(A_deflated, x)
            except np.linalg.LinAlgError:
                raise ValueError(f"La matriz deflactada es singular en el valor propio {idx + 1}.")

            # Normalizar
            y = y / np.linalg.norm(y)

            # Calcular el valor propio usando el cociente de Rayleigh
            lambda_current = np.dot(y, A @ y) / np.dot(y, y)

            # Criterio de convergencia
            if abs(lambda_current - lambda_prev) < tolerancia:
                eigenvalues.append(lambda_current)
                eigenvectors.append(y)
                iterations.append(k)

                # Deflación: eliminar el efecto del valor propio encontrado
                # A_deflated = A_deflated - λ * (v * v^T)
                v = y.reshape(-1, 1)
                A_deflated = A_deflated - lambda_current * (v @ v.T)

                break

            lambda_prev = lambda_current
            x = y.copy()
        else:
            raise ValueError(f"Método de la potencia inversa no convergió para el valor propio {idx + 1} en {max_iter} iteraciones.")

    return np.array(eigenvalues), eigenvectors, iterations


def demo_metodo_potencia_inversa():
    """
    Demostración del método de la potencia inversa para calcular todos los valores y vectores propios.
    Compara el resultado con la función eig de numpy.linalg.
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=100)

    print("\n" + "="*80)
    print("DEMOSTRACIÓN: MÉTODO DE LA POTENCIA INVERSA")
    print("="*80)

    # Matriz dada en el problema
    A = np.array([
        [17,  1,   2],
        [ 1, 14,  -2],
        [ 2, -2,  20]
    ], dtype=float)

    print("\nMatriz A:")
    print(A)

    try:
        eigenvalues, eigenvectors, iterations = potencia_inversa(A, tolerancia=1e-7, max_iter=1000)
        
        print("\nMétodo de la Potencia Inversa:")
        print("-" * 80)
        for i, (lam, vec, k) in enumerate(zip(eigenvalues, eigenvectors, iterations)):
            print(f"\nValor propio {i+1} (en {k} iteraciones): {lam:.10f}")
            print(f"Vector propio asociado:")
            print(vec)

        # Comparación con numpy.linalg.eig
        print("\n" + "="*80)
        print("COMPARACIÓN CON numpy.linalg.eig")
        print("="*80)
        
        eigenvalues_np, eigenvectors_np = np.linalg.eig(A)
        
        # Ordenar los valores propios de numpy por valor descendente
        idx_sorted = np.argsort(eigenvalues_np)[::-1]
        eigenvalues_np = eigenvalues_np[idx_sorted]
        eigenvectors_np = eigenvectors_np[:, idx_sorted]

        print("\nnumpy.linalg.eig:")
        print("-" * 80)
        for i in range(len(eigenvalues_np)):
            print(f"\nValor propio {i+1}: {eigenvalues_np[i].real:.10f}")
            print(f"Vector propio asociado:")
            vec_np = eigenvectors_np[:, i].real
            vec_np = vec_np / np.linalg.norm(vec_np)
            print(vec_np)

        # Calcular diferencias
        print("\n" + "="*80)
        print("DIFERENCIAS ENTRE MÉTODOS")
        print("="*80)
        
        for i in range(len(eigenvalues)):
            diff_eigenvalue = abs(eigenvalues[i] - eigenvalues_np[i].real)
            
            # Normalizar y ajustar signo del vector propio de numpy
            vec_np = eigenvectors_np[:, i].real
            vec_np = vec_np / np.linalg.norm(vec_np)
            
            if np.dot(eigenvectors[i], vec_np) < 0:
                vec_np = -vec_np
            
            diff_eigenvector = np.linalg.norm(eigenvectors[i] - vec_np)
            
            print(f"\nValor propio {i+1}:")
            print(f"  Diferencia en valor propio: {diff_eigenvalue:.2e}")
            print(f"  Diferencia en vector propio (norma): {diff_eigenvector:.2e}")

    except ValueError as e:
        print(f"\nError: {e}")

    # --- Caso adicional: Matriz diagonal ---
    print("\n" + "="*80)
    print("CASO ADICIONAL: Matriz diagonal (caso simple)")
    print("="*80)

    A2 = np.array([
        [5, 0, 0],
        [0, 3, 0],
        [0, 0, 1]
    ], dtype=float)

    print("\nMatriz A:")
    print(A2)

    try:
        eigenvalues2, eigenvectors2, iterations2 = potencia_inversa(A2, tolerancia=1e-7, max_iter=1000)
        
        print("\nMétodo de la Potencia Inversa:")
        print("-" * 80)
        for i, (lam, vec, k) in enumerate(zip(eigenvalues2, eigenvectors2, iterations2)):
            print(f"\nValor propio {i+1} (en {k} iteraciones): {lam:.10f}")
            print(f"Vector propio: {vec}")

        print("\nValores propios esperados (diagonal): [5, 3, 1]")
        
    except ValueError as e:
        print(f"\nError: {e}")


def demo():
    demo_metodo_potencia_inversa()

demo()