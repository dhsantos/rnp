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


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]






class configuracion:
    def __init__(self):
        #--------------------------------------------
        #------Parametros
        self.imagen = "contents/marian.jpeg"
        self.estilo = "styles/retrato2.jpeg"
        self.redConv = "imagenet-vgg-verydeep-19.mat"
        self.alfa = 20
        self.beta = 80
        self.iteraciones = 150
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








def content_loss_func(sess, model):

    def _content_loss(p, x):

        N = p.shape[3]

        M = p.shape[1] * p.shape[2]
    

        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])



def style_loss_func(sess, model):

    def _gram_matrix(F, N, M):
        
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        
        N = a.shape[3]

        M = a.shape[1] * a.shape[2]

        A = _gram_matrix(a, N, M)

        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss



def correr(conf):
    alpha = conf.alfa
    beta = conf.beta

    model = load_vgg_model(conf.redConv)

    content_image = scipy.misc.imread(conf.imagen)
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread(conf.estilo)
    style_image = reshape_and_normalize_image(style_image)



    generated_image = generate_noise_image(content_image)


    sess.run(tf.initialize_all_variables())

    sess.run(model['input'].assign(content_image))
    content_loss = content_loss_func(sess, model)

    sess.run(model['input'].assign(style_image))
    style_loss = style_loss_func(sess, model)

    total_loss = beta * content_loss + alpha * style_loss

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)


    model_nn(sess, generated_image, conf.iteraciones, model, train_step, total_loss, content_loss, style_loss)


def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, [1, n_H*n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [1, n_H*n_W, n_C])

    J_content = (1/(4*n_H*n_W*n_C))*(tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))))
    
    return J_content


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.reshape(a_S, [n_C, n_H*n_W])
    a_G = tf.reshape(a_G, [n_C, n_H*n_W])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = (1/(4*(n_C**2)*(n_H*n_W)**2))*(tf.reduce_sum(tf.square(tf.subtract(GS, GG))))
        
    return J_style_layer




def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]

        a_S = sess.run(out)

        a_G = out

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style



def total_cost(J_content, J_style, alpha, beta):
    J = alpha*J_content + beta*J_style
    
    return J


def model_nn(sess, input_image, num_iterations, model, train_step, J, J_content, J_style):
    sess.run(tf.initialize_all_variables())

    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        _ = sess.run(train_step)
        generated_image = sess.run(model['input'])

        if i%10 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            save_image("iteracion - " + str(i) + ".png", generated_image)
    
    save_image('resultado.jpg', generated_image)
    
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
