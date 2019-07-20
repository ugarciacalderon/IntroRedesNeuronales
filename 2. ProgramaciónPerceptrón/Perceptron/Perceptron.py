import numpy as np


class Perceptron:
    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

    def Entrenamiento(self):
        """
        Este método tiene como objetivo simular el proceso de un perceptrón, dentro de sus etapas encontramos:
        1. Genera pesos aleatorios con las misma dimensiones de la matriz de entradas
        2. Realiza suma poderada de entradas con pesos
        3. Se aplica la función de activación para obtener una salida y, y = 0 si la suma < 0 sino y = 1
        4. Se compara la salida y con la salida esperada si son iguales se procede con el siguiente vector de entrada,
        de lo contrario se generan nuevos pesos aleatorios
        :return:
        """
        epochs, contador = 0, 0

        while contador < 4:
            print('---------- epochs {} ---------- '.format(epochs))
            # se generan pesos aleatorios en el rango [-1,1]
            weights = np.array(np.random.uniform(-1, 1, self.inputs.shape))
            for input,weight, output in zip(self.inputs, weights, self.outputs):
                # Realiza la suma ponderada de entradas con pesos
                salida_generada = input@weight

                # Función sigmoide
                salida_generada = 0 if salida_generada < 0 else 1

                if salida_generada == output:
                    contador +=1
                else:
                    contador = 0

                print('entrada: ', input, 'pesos:', weight, 'salida_esperada: ', output, 'salida_obtenida: ',
                      salida_generada)

            epochs +=1

