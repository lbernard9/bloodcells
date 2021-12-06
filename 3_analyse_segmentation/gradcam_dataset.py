import matplotlib.cm as cm
import tensorflow as tf
from keras import preprocessing
from tensorflow.keras.models import Model
from IPython.display import Image, display
import numpy as np
import matplotlib.pyplot as plt




def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )


    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(tf.expand_dims(img_array, axis=0))
        pred_index_model = tf.argmax(preds[0])
        if pred_index is None:
            pred_index = pred_index_model
        else:
            if pred_index!=pred_index_model:
                print("Cellule non prédite dans la catégorie attendue")
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_headmap(preprocess_function, img_path, model, last_conv_layer_name, pred_index=None):
    img_array = preprocess_function(img_path)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    
    superimposed_img = get_gradcam(img_path,heatmap)
    
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    # Display heatmap
    fig,axs = plt.subplots(1,3,figsize=(8,5))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[1].imshow(heatmap)
    axs[1].axis('off')
    axs[2].imshow(superimposed_img)
    axs[2].axis('off')
    plt.show()


def get_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = preprocessing.image.load_img(img_path)
    img = preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img