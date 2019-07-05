import numpy as np


class Perceptron:
    def __init__(self):
        pass

    def Funcionamiento(self, inputs, outputs):
        epochs, contador = 0, 0

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        while contador < 4:
            weights = np.array(np.random.uniform(-1, 1, inputs.shape))

            for input,weight, output in zip(inputs, weights, outputs):
                # Realiza la suma ponderada de entradas con pesos
                salida_generada = 0 if (input@weight) < 0 else 1

                print('entrada: ', input, 'pesos:', weight, 'salida_esperada: ', output, 'salida_obtenida: ',
                      salida_generada)

                if salida_generada == output:
                    contador +=1
                else:
                    contador = 0

            epochs +=1
            print('epochs: ', epochs)

            #print('entrada: ',input, 'pesos:' ,weight, 'salida_esperada: ',output, 'salida_obtenida: ', saliada_generada)



