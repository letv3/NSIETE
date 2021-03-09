import numpy as np

def gradient_check_n(network, criterion, X, Y, epsilon=1e-7):
    # Set-up variables
    gradapprox = []
    grad_backward = []

    for layer in network:
        # Compute gradapprox
        if not hasattr(layer, "W"):
            continue
        shape = layer.W.shape
        # print(shape[0], ',', shape[1])
        for i in range(shape[0]):
            for j in range(shape[1]):
                # print('i',i,'j',j)
                # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
                # "_" is used because the function you have to outputs two parameters but we only care about the first one
                origin_W = np.copy(layer.W.data[i][j])

                layer.W.data[i][j] = origin_W + epsilon
                A_plus = network(X)
                J_plus = criterion(A_plus, Y).data

                # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
                layer.W.data[i][j] = origin_W - epsilon
                A_minus = network(X)
                J_minus = criterion(A_minus, Y).data

                # Compute gradapprox[i]
                gradapprox.append((J_plus - J_minus) / (2 * epsilon))
                # print(layer.name, layer.dW.shape)
                # grad = np.mean(layer.dW, axis=0, keepdims=True)
                # grad_backward.append(grad[0][i][j])
                grad_backward.append(layer.W.grad[i][j])
                layer.W.data[i][j] = origin_W

    # Compare gradapprox to backward propagation gradients by computing difference.
    gradapprox = np.reshape(gradapprox, (-1, 1))
    grad_backward = np.reshape(grad_backward, (-1, 1))

    numerator = np.linalg.norm(grad_backward - gradapprox)
    denominator = np.linalg.norm(grad_backward) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7 or not difference:
        print(
            "\033[91m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")


def softmax(inputs, *args, **kwargs):
    exps = np.exp(inputs)
    return exps / np.sum(exps, axis=0)


def identity(inputs, *args, **kwargs):
    return inputs
