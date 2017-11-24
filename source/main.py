import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

class configuracion:
    def __init__(self):
        #--------------------------------------------
        #------Parametros
        self.imagen = "contents/casa.jpg"
        self.estilo = "styles/picasso.jpg"
        self.redConv = "imagenet-vgg-verydeep-19.mat"
        self.alfa = 40
        self.beta = 60
        self.iteraciones = 1000
        self.learningRate = 2.0

    def cambiar_valor(self, clave, valor):
        if clave == 'alfa':
            conf.alfa = int(valor)
            conf.beta = 100 - conf.alfa
        elif clave == 'beta':
            conf.beta = int(valor)
            conf.alfa = 100 - conf.beta
        elif clave == 'r':
            conf.redConv = valor
        elif clave == 'c':
            conf.imagen = valor
        elif clave == 's':
            conf.estilo = valor
        elif clave == 'i':
            conf.iteraciones = int(valor)
        elif clave == 'lr':
            conf.learningRate = float(valor)


def mostrar_menu(conf, msg_pantalla):
    os.system('clear');
    print ("MENU PRINCIPAL:\n")
    print ("Configuracion de parametros:\n")
    print ("  - imagen                     (c)           " + str(conf.imagen))
    print ("  - imagen de estilo           (s)           " + str(conf.estilo))
    print ("  - red convolucional          (r)           " + str(conf.redConv))
    print ("  - peso imagen                (alfa)        " + str(conf.alfa))
    print ("  - peso estilo                (beta)        " + str(conf.beta))
    print ("  - cantidad iteraciones       (i)           " + str(conf.iteraciones))
    print ("  - learning rate              (lr)          " + str(conf.learningRate))
    print ("\n")
    print ("run  -> para ejecutar el programa")
    print ("help -> para ver los comandos validos y su modo de uso")
    print ("exit -> terminar la ejecucion del programa\n")
    if len(msg_pantalla) > 0:
        print ("Respuesta:  "+str(msg_pantalla))
    else:
        print ("")
    print ("")

def mostrar_ayuda():
    os.system('clear');
    print ("AYUDA:\n")
    print (" Tipo de datos")
    print ("         alfa: valor entero entre 0 - 100")
    print ("         beta: valor entero entre 0 - 100")
    print ("         learning rate: valor decimal")
    print ("         contidad iteraciones: valor entero se sugiere mayor a 1000")
    print ("         red convolucional: red preentrenada, recomendada: imagenet-vgg-verydeep-19, si se modifica se debe cambiar las layer usadas a mano en el codigo y volver a ejecutar")
    print (" Cambiar parametro")
    print ("         Descripcion: modifica el valor de un parametro")
    print ("         Uso: change codigo_parametro nuevo_valor")
    print ("         Ejemplo: change alfa 50")
    print ("")



def correr(conf):
    # Start interactive session

    alpha = conf.alfa
    beta = conf.beta

    model = load_vgg_model(conf.redConv)

    content_image = scipy.misc.imread(conf.imagen)
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread(conf.estilo)
    style_image = reshape_and_normalize_image(style_image)

    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]

    generated_image = generate_noise_image(content_image)

    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # **Note**: At this point, a_G is a tensor and hasn't been evaluated. It will be evaluated and updated at each iteration when we run the Tensorflow graph in model_nn() below.
    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS)

    J = total_cost(J_content, J_style,  alpha = alpha, beta = beta)

    optimizer = tf.train.AdamOptimizer(conf.learningRate)
    train_step = optimizer.minimize(J)

    model_nn(sess, generated_image, conf.iteraciones, model, train_step, J, J_content, J_style)


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (2 lines)
    a_C_unrolled = tf.reshape(a_C, [1, n_H*n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [1, n_H*n_W, n_C])

    # compute the cost with tensorflow (1 line)
    J_content = (1/(4*n_H*n_W*n_C))*(tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))))
    ### END CODE HERE ###
    
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    ### START CODE HERE ### (1 line)
    GA = tf.matmul(A, tf.transpose(A))
    ### END CODE HERE ###
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (2 lines)
    a_S = tf.reshape(a_S, [n_C, n_H*n_W])
    a_G = tf.reshape(a_G, [n_C, n_H*n_W])

    # Computing gram_matrices for both images S and G (2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (1 line)
    J_style_layer = (1/(4*(n_C**2)*(n_H*n_W)**2))*(tf.reduce_sum(tf.square(tf.subtract(GS, GG))))
    
    ### END CODE HERE ###
    
    return J_style_layer




def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style



def total_cost(J_content, J_style, alpha, beta):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    ### START CODE HERE ### (1 line)
    J = alpha*J_content + beta*J_style
    ### END CODE HERE ###
    
    return J


def model_nn(sess, input_image, num_iterations, model, train_step, J, J_content, J_style):
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        _ = sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%50 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            save_image(str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('generated_image.jpg', generated_image)
    
    return generated_image


conf = configuracion()
salir = 0
msg_pantalla = ""
sess = tf.InteractiveSession()


while (not salir):
    mostrar_menu(conf, msg_pantalla)
    msg_pantalla = ""
    comando = input("Ingrese un comando: ");
    comando = comando.split(" ")
    if (comando[0] == 'exit'):
        salir = 1
    elif (comando[0] == 'help'):
        mostrar_ayuda()
        input("Pulse enter para volver al menu...")
    elif (comando[0] == 'change'):
        conf.cambiar_valor(comando[1], comando[2])
    elif (comando[0] == 'run'):
        correr(conf);
        input("Pulse enter para volver al menu...")
