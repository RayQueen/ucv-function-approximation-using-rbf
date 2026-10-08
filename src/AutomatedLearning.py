import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================
#             Generación de Datos
# =============================================
# Genera un conjunto de n puntos aleatorios en el plano [limInf, limSup]^2 usando una función no lineal y añade un ruido gaussiano pequeño
def genData(n, limInf, limSup, noise_std=0.1):
    # Crea n puntos aleatorios (x, y) bajo una distribución uniforme en el rango [limInf, limSup]
    x = np.random.uniform(limInf, limSup, n)
    y = np.random.uniform(limInf, limSup, n)

    # Calcula z como una función no lineal de (x, y) y añade ruido gaussiano con noise_std como desviación estándar
    z = np.sin(x) * np.cos(y) + noise_std * np.random.normal(size=n)

    return x, y, z

# =============================================
#          Implementación del modelo
# =============================================
# Construye la matriz de kernel gaussiano (Phi) para los puntos (x, y) y un parámetro de forma epsilon
def buildKernelMatrix(x, y, epsilon):
    n = len(x)
    Phi = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Calcula la distancia euclidiana al cuadrado entre los puntos (x[i], y[i]) y (x[j], y[j])
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            # Calcula el valor del kernel gaussiano para la distancia d2 y el parámetro de forma epsilon
            Phi[i, j] = np.exp(-(epsilon ** 2) * d2)

    return Phi

# Resuelve el sistema regularizado para obtener w
def solveRegularizedSystem(Phi, y, lambdaReg):
    n = len(y)
    A = Phi + lambdaReg * np.eye(n)
    w = np.linalg.solve(A, y)
    return w

# Evalúa el modelo en un punto (x0, y0) dado los datos de entrenamiento (xData, yData), los pesos w y el parámetro de forma epsilon
def evaluatePoint(x0, y0, xData, yData, w, epsilon):
    return np.sum(w * np.exp(-(epsilon ** 2) * ((x0 - xData) ** 2 + (y0 - yData) ** 2)))

# =============================================
#       Visualización de la superficie
# =============================================
# Evalúa el modelo en una malla fina (50x50) y grafica la superficie de predicción
# Si se pasa file_name, guarda la figura con ese nombre y un encabezado textual

def visualizeSurface(x, y, z, w, epsilon, file_name=None, title=None, show=True):
    # Crear una malla fina para la visualización
    x_fine = np.linspace(x.min(), x.max(), 50)
    y_fine = np.linspace(y.min(), y.max(), 50)
    X, Y = np.meshgrid(x_fine, y_fine)

    # Evaluar el modelo en cada punto de la malla
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = evaluatePoint(X[i, j], Y[i, j], x, y, w, epsilon)

    # Crear la figura y el eje 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.8, cmap='viridis', edgecolor='none')
    ax.scatter(x, y, z, c='r', s=25, label='Datos de entrenamiento (con ruido)')

    heading = title if title is not None else f'epsilon = {epsilon:.3f}'
    ax.set_title(heading)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    if file_name is not None:
        folder = os.path.dirname(file_name)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        fig.savefig(file_name, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return X, Y, Z

# =============================================
#       Estudio del Parámetro de Forma
# =============================================
if __name__ == "__main__":
    # Generar datos de prueba dentro del rango [-2, 2]
    x, y, z = genData(100, -2, 2)

    # Definir el parámetro de regularización y los valores de epsilon a estudiar
    lambdaReg = 1e-6
    epsilon = 1.5
    epsilons = [0.05 * epsilon, epsilon, 5 * epsilon]
    computed_surfaces = {}

    # Ruta de salida para guardar las imágenes generadas
    output_dir = Path(__file__).resolve().parent.parent / 'img'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualizar la superficie para cada valor de epsilon y guardar las imágenes
    for epsilon in epsilons:
        Phi = buildKernelMatrix(x, y, epsilon)
        w = solveRegularizedSystem(Phi, z, lambdaReg)
        small, optimal, large = visualizeSurface(
            x, y, z, w, epsilon,
            file_name=os.path.join(output_dir, f'epsilon_{epsilon:.2f}.png'),
            title=f'epsilon = {epsilon}',
            show=False
        )
        computed_surfaces[epsilon] = (small, optimal, large)

    # Crear una figura con tres subplots para mostrar las superficies
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 6))
    for ax, epsilon in zip(axes, epsilons):
        small, optimal, large = computed_surfaces[epsilon]
        ax.plot_surface(small, optimal, large, alpha=0.8, cmap='viridis', edgecolor='none')
        ax.scatter(x, y, z, c='r', s=20, label='Datos de entrenamiento (con ruido)')
        ax.set_title(f'epsilon = {epsilon:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()

    plt.subplots_adjust(wspace=0.0)
    fig.suptitle('Comparación de los valores de epsilon', fontsize=16)
    plt.tight_layout()
    plt.show()

