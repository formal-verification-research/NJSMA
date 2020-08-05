"""
This tutorial shows how to generate adversarial examples
using JSMA in white-box setting.
The original paper can be found at:
https://arxiv.org/abs/1511.07528
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans.train import train
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
from cleverhans_tutorials.tutorial_models import make_basic_picklable_cnn
from cleverhans.serial import save
from cleverhans.serial import load

from random import seed
from random import randint

FLAGS = flags.FLAGS

VIZ_ENABLED = True
NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
SOURCE_SAMPLES = 500
TRAIN_NEW = 0
USE_MOD = 0
SEED = 69

#print("\n\n\n\n\n\n\n", SaliencyMapMethod.it_count, "\n\n\n\n\n")

def mnist_tutorial_jsma(train_start=0, train_end=60000, test_start=0,
                        test_end=10000, viz_enabled=VIZ_ENABLED,
                        nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                        source_samples=SOURCE_SAMPLES,
                        learning_rate=LEARNING_RATE):
  """
  MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param viz_enabled: (boolean) activate plots of adversarial examples
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param nb_classes: number of output classes
  :param source_samples: number of test inputs to attack
  :param learning_rate: learning rate for training
  :return: an AccuracyReport object
  """
  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session and set as Keras backend session
  #replace
  num_threads = None
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))
  #with sess = tf.Session()
  print("Created TensorFlow session.")

  set_log_level(logging.DEBUG)

  # Get MNIST test data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]
  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  nb_filters = 64
  
  # Define TF model graph
  model = make_basic_picklable_cnn()
                  
  preds = model.get_logits(x)
  loss = CrossEntropy(model, smoothing=0.1)
  print("Defined TensorFlow model graph.")

  ###########################################################################
  # Training the model using TensorFlow
  ###########################################################################

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  dataset = tf.data.Dataset.from_tensor_slices((tf.reshape(x_train, [60000, 28, 28]), y_train))
  dataset = dataset.batch(32)
  val_dataset = tf.data.Dataset.from_tensor_slices((tf.reshape(x_test, [10000, 28, 28]), y_test))
  val_dataset = val_dataset.batch(32)

  
  sess.run(tf.global_variables_initializer())
  rng = np.random.RandomState([2017, 8, 30])
  if TRAIN_NEW == 1:
    with sess.as_default():
        train(sess, loss, x_train, y_train, args=train_params, rng=rng)
        save("test.joblib", model)
  else:
    with sess.as_default():
        model = load("test.joblib")#changed
    assert len(model.get_params()) > 0
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)

  # Evaluate the accuracy of the MNIST model on legitimate test examples
  eval_params = {'batch_size': batch_size}
  accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
  assert x_test.shape[0] == test_end - test_start, x_test.shape
  print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
  report.clean_train_clean_eval = accuracy

  ###########################################################################
  # Craft adversarial examples using the Jacobian-based saliency map approach
  ###########################################################################
  print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes - 1) +
        ' adversarial examples')

  # Keep track of success (adversarial example classified in target)
  results = np.zeros((nb_classes, source_samples), dtype='i')

  # Rate of perturbed features for each test set example and target class
  perturbations = np.zeros((nb_classes, source_samples), dtype='f')

  # Initialize our array for grid visualization
  grid_shape = (nb_classes, nb_classes, img_rows, img_cols, nchannels)
  grid_viz_data = np.zeros(grid_shape, dtype='f')
    
  # Instantiate a SaliencyMapMethod attack object
  jsma = SaliencyMapMethod(model, sess=sess)
  jsma_params = {'theta': 1., 'gamma': 0.1,
                 'clip_min': 0., 'clip_max': 1.,
                 'y_target': None}

  figure = None
  # Loop over the samples we want to perturb into adversarial examples
  seed(SEED)
  for sample_ind in xrange(0, source_samples):
    img = randint(0, 10000)
    print('--------------------------------------')
    print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
    sample = x_test[img: (img + 1)]#sample = x_test[sample_ind:(sample_ind + 1)]

    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset)
    current_class = int(np.argmax(y_test[img])) #current_class = int(np.argmax(y_test[sample_ind]))
    target_classes = other_classes(nb_classes, current_class)

    # For the grid visualization, keep original images along the diagonal
    grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
        sample, (img_rows, img_cols, nchannels))
    tn = 0
    totc = 0
    # Loop over all target classes
    for target in target_classes:
      print('Generating adv. example for target class %i' % target)

      # This call runs the Jacobian-based saliency map approach
      one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
      one_hot_target[0, target] = 1
      jsma_params['y_target'] = one_hot_target
      adv_x = jsma.generate_np(sample, **jsma_params)

      # Check if success was achieved
      res = int(model_argmax(sess, x, preds, adv_x) == target)

      # Compute number of modified features
      adv_x_reshape = adv_x.reshape(-1)
      test_in_reshape = x_test[sample_ind].reshape(-1)
      nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
      percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]
      diff = np.array(adv_x-sample)
      #print(np.sum(diff))
      diff = np.reshape(diff,(28,28))
      diff = diff*255
      cv2.imwrite("test.png",diff)
      diff = cv2.imread("test.png")
      diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
      nieghbors = 0
      tc = 0
      for i in range(0,28,1):
        for j in range(0,28,1):
            if diff[i,j] > 0:
                tc = tc + 1
                totc = totc + 1
                if i >0 and i <27 and j>0 and j<27:#main grid not edges or corners
                    if diff[i-1, j-1] >0:
                        nieghbors = nieghbors +1
                    if diff[i-1, j] >0:
                        nieghbors = nieghbors +1
                    if diff[i-1, j+1] >0:
                        nieghbors = nieghbors +1
                    if diff[i, j-1] >0:
                        nieghbors = nieghbors +1
                    if diff[i, j+1] >0:
                        nieghbors = nieghbors +1
                    if diff[i+1, j-1] >0:
                        nieghbors = nieghbors +1
                    if diff[i+1, j] >0:
                        nieghbors = nieghbors +1
                    if diff[i+1, j+1] > 0:
                        nieghbors = nieghbors +1
                else:
                    #corners
                    if i ==0 and j ==0:
                        if diff[i,j+1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i+1,j] >0:
                            nieghbors = nieghbors +1
                    if i ==27 and j ==0:
                        if diff[i,j+1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i-1,j] >0:
                            nieghbors = nieghbors +1
                    if i ==0 and j ==27:
                        if diff[i,j-1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i+1,j] >0:
                            nieghbors = nieghbors +1
                    if i ==27 and j ==27:
                        if diff[i,j-1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i-1,j] >0:
                            nieghbors = nieghbors +1
                    #edges
                    if i == 0 and j >0 and j< 27:#left side
                        if diff[i,j-1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i,j+1] >0:
                            nieghbors = nieghbors +1
                        if diff[i + 1, j-1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i+1, j] >0:
                            nieghbors = nieghbors +1
                        if diff[i+1, j+1] > 0:
                            nieghbors = nieghbors +1
                    if i == 27 and j >0 and j< 27:#right side
                        if diff[i,j-1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i,j+1] >0:
                            nieghbors = nieghbors +1
                        if diff[i - 1, j-1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i-1, j] >0:
                            nieghbors = nieghbors +1
                        if diff[i-1, j+1] > 0:
                            nieghbors = nieghbors +1
                    if j == 0 and i >0 and i< 27:#top side
                        if diff[i-1,j] > 0:
                            nieghbors = nieghbors +1
                        if diff[i+1,j] >0:
                            nieghbors = nieghbors +1
                        if diff[i -1, j+1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i, j+1] >0:
                            nieghbors = nieghbors +1
                        if diff[i+1, j+1] > 0:
                            nieghbors = nieghbors +1
                    if j == 27 and i >0 and i< 27:#bot side
                        if diff[i-1,j] > 0:
                            nieghbors = nieghbors +1
                        if diff[i+1,j] >0:
                            nieghbors = nieghbors +1
                        if diff[i - 1, j-1] > 0:
                            nieghbors = nieghbors +1
                        if diff[i, j-1] >0:
                            nieghbors = nieghbors +1
                        if diff[i+1, j-1] > 0:
                            nieghbors = nieghbors +1
      
      # print(tc)      
      # print(nieghbors)
      tn = tn + nieghbors
      # if tc > 0:
        # print(nieghbors/tc)
      # Display the original and adversarial images side-by-side
      if viz_enabled:
        figure = pair_visual(
            np.reshape(sample, (img_rows, img_cols, nchannels)),
            np.reshape(adv_x, (img_rows, img_cols, nchannels)), figure)
      # Add our adversarial example to our grid data
      grid_viz_data[target, current_class, :, :, :] = np.reshape(
          adv_x, (img_rows, img_cols, nchannels))

      # Update the arrays for later analysis
      results[target, sample_ind] = res
      perturbations[target, sample_ind] = percent_perturb
      #print(perturbations[target, sample_ind])
      
  print('--------------------------------------')
  
  print("average neighbors per modified pixel ", tn/totc)  
  # Compute the number of adversarial examples that were successfully found
  nb_targets_tried = ((nb_classes - 1) * source_samples)
  succ_rate = float(np.sum(results)) / nb_targets_tried
  print('Avg. rate of successful adv. examples {0:.8f}'.format(succ_rate))
  report.clean_train_adv_eval = 1. - succ_rate

  # Compute the average distortion introduced by the algorithm
  percent_perturbed = np.mean(perturbations)
  
  s = perturbations.shape
  myPert = np.empty(0)
  myResults = np.empty(0)
  for i in range(s[0]):
    for j in range(s[1]):
      if perturbations[i][j] > 0:
        myPert = np.append(myPert, perturbations[i][j])
        myResults = np.append(myResults, results[i][j])
  min_perturbed = np.min(myPert)
  max_perturbed = np.max(myPert)
  
  s2 = myResults.shape
  final = np.empty(0)
  for i in range(s2[0]):
    if myResults[i]>0:
        final = np.append(final, myPert[i])
  
  print('Avg. rate of perturbed features {0:.8f}'.format(percent_perturbed))
  print('MIN of perturbed features {0:.8f}'.format(min_perturbed))
  print('MAX of perturbed features {0:.8f}'.format(max_perturbed))

  # Compute the average distortion introduced for successful samples only
  percent_perturb_succ = np.mean(perturbations * (results == 1))
  min_perturb_succ = np.min(final)
  max_perturb_succ = np.max(final)
  print('Avg. rate of perturbed features for successful '
        'adversarial examples {0:.8f}'.format(percent_perturb_succ))
  print('Min of perturbed features for successful '
        'adversarial examples {0:.8f}'.format(min_perturb_succ))
  print('Max of perturbed features for successful '
        'adversarial examples {0:.8f}'.format(max_perturb_succ))
  
  #Close TF session
  sess.close()

  # Finally, block & display a grid of all the adversarial examples
  if viz_enabled:
    import matplotlib.pyplot as plt
    plt.close(figure)
    _ = grid_visual(grid_viz_data)

  return report


def main(argv=None):
  
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial_jsma(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
  flags.DEFINE_boolean('viz_enabled', VIZ_ENABLED,
                       'Visualize adversarial ex.')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('source_samples', SOURCE_SAMPLES,
                       'Nb of test inputs to attack')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  tf.app.run()
