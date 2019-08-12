# -*- coding: utf-8 -*-
"""Neural style transfer

Original file is located at
    https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/style_transfer.ipynb
This code is taken from the TensorFlow tutorial - all rights for the code belong to the respective authors of that code
"""

################################################################################
#                                                                              #
#                         Import and Configure Modules                         #
#                                                                              #
################################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools
#%%
################################################################################
#                                                                              #
#                               Download Images                                #
#                                                                              #
################################################################################

content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

################################################################################
#                                                                              #
#                             Visualize the Input                              #
#                                                                              #
################################################################################
"""## Visualize the input | Define a function to load an image and limit its
maximum dimension to 512 pixels.  """

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

"""Create a simple function to display an image:"""

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

#%%
"""## Define content and style representations """

x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

################################################################################
#                                                                              #
#            Load the pretrained model VGG19 and create the models             #
#                                                                              #
################################################################################
"""Now load a `VGG19` without the classification head, and list the layer names"""

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# print()
# for layer in vgg.layers:
#   print(layer.name)

#%%

"""Choose intermediate layers from the network to represent the style and
content of the image:"""

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

"""#### This following function builds a VGG19 model that returns a list of
intermediate layer outputs: """

def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model
#%%

"""And to create the model:"""

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
# for name, output in zip(style_layers, style_outputs):
#   print(name)
#   print("  shape: ", output.numpy().shape)
#   print("  min: ", output.numpy().min())
#   print("  max: ", output.numpy().max())
#   print("  mean: ", output.numpy().mean())
#   print()
#%%

################################################################################
#                                                                              #
#            Load the pretrained model VGG19 and create the models             #
#                                                                              #
################################################################################

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

"""## Extract style and content Build a model that returns the style and
content tensors.  """

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}

"""When called on an image, this model returns the gram matrix (style) of the
`style_layers` and content of the `content_layers`:"""

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

# print('Styles:')
# for name, output in sorted(results['style'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())
#   print()

# print("Contents:")
# for name, output in sorted(results['content'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())

#%%
################################################################################
#                                                                              #
#                             Run Gradient Descent                             #
#                                                                              #
################################################################################
"""## Run gradient descent

With this style and content extractor, you can now implement the style transfer
algorithm. Do this by calculating the mean square error for your image's output
relative to each target, then take the weighted sum of these losses.

Set your style and content target values:
"""

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

"""Define a `tf.Variable` to contain the image to optimize. To make this quick,
initialize it with the content image (the `tf.Variable` must be the same shape
        as the content image):"""

image = tf.Variable(content_image)

"""Since this is a float image, define a function to keep the pixel values
between 0 and 1:"""

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

"""Create an optimizer. The paper recommends LBFGS, but `Adam` works okay, too:"""

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

"""To optimize this, use a weighted combination of the two losses to get the
total loss:"""

style_weight=1e-2
content_weight=1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

"""Use `tf.GradientTape` to update the image."""

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

"""Now run a few steps to test:"""

train_step(image)
train_step(image)
train_step(image)
plt.imshow(image.read_value()[0])
#%%


################################################################################
#                                                                              #
#                         Perform Larger Optimization                          #
#                                                                              #
################################################################################


"""Since it's working, perform a longer optimization:"""

import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')
  display.clear_output(wait=True)
  imshow(image.read_value())
  plt.title("Train step: {}".format(step))
  plt.show()

end = time.time()
print("Total time: {:.1f}".format(end-start))

#%%

################################################################################
#                                                                              #
#           Reduce the total variation loss (the noise in the image)           #
#                                                                              #
################################################################################


"""## Total variation loss

One downside to this basic implementation is that it produces a lot of high
frequency artifacts. Decrease these using an explicit regularization term on
the high frequency components of the image. In style transfer, this is often
called the *total variation loss*:
"""

def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var

x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2,2,2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2,2,3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2,2,4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")

"""This shows how the high frequency components have increased.  """

# plt.figure(figsize=(14,10))

# sobel = tf.image.sobel_edges(content_image)
# plt.subplot(1,2,1)
# imshow(clip_0_1(sobel[...,0]/4+0.5), "Horizontal Sobel-edges")
# plt.subplot(1,2,2)
# imshow(clip_0_1(sobel[...,1]/4+0.5), "Vertical Sobel-edges")

"""The regularization loss asociated with this is the sum of the squares of the
values:"""

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)


################################################################################
#                                                                              #
#                            Rerun the optimization                            #
#                                                                              #
################################################################################


"""## Re-run the optimization

Choose a weight for the `total_variation_loss`:
"""

total_variation_weight=1e8

"""Now include it in the `train_step` function:"""

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*total_variation_loss(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

"""Re-initialise the optimization variable:"""

image = tf.Variable(content_image)

"""And run the optimization:"""
#%%
import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')
  display.clear_output(wait=True)
  imshow(image.read_value())
  plt.title("Train step: {}".format(step))
  plt.show()

end = time.time()
print("Total time: {:.1f}".format(end-start))


################################################################################
#                                                                              #
#                               Save the result                                #
#                                                                              #
################################################################################

file_name = 'kadinsky-turtle.png'
mpl.image.imsave(file_name, image[0])

# files.download(file_name)
