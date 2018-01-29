##########################################################
# A generic datatool for loading jpegs and csv labels
#   built for the Intel Movidius Competition
# copyright shadySource
# MIT Liscense
##########################################################
import random
from os.path import join
from PIL import Image
import numpy as np

random.seed(173)

debug = False

data_dir = join('..', 'data')
train_image_dir = join(data_dir, 'training')
prov_image_dir = join(data_dir, 'provisional')
csv_path = join(data_dir, 'training_ground_truth.csv')

with open(csv_path, 'r') as csv:
    # read the csv file
    labels = csv.read()
    labels = [i.split(',')[:-1] for i in labels.split('\n')[1:-1]]
    # turn labels frm strings '123' into ints 123
    labels = [(i[0], int(i[1])-1) for i in labels]

if debug:
    print('First label is', labels[0])

num_val = int(.005*len(labels))
val = labels[-num_val:]
assert num_val == len(val)

num_test = num_val
test = labels[-2*num_val:-num_val]
assert num_test == len(test)

num_train = len(labels) - 2*num_val
train = labels[:num_train]
assert num_train == len(train)

if debug:
    print('There are', len(labels), 'labels.')
    print('Reserving', num_train, 'labels for trainin')
    print('Reserving', num_test, 'labels for validation')
    print('Reserving', num_val, 'labels for testing')

# explore number of each category(they are all 400 samples)
# label_baskets = [0]*200
# for label in labels:
#     label_baskets[label[1]-1] += 1
# print(max(label_baskets))
# print(min(label_baskets))

if debug:
    im = Image.open(join(train_image_dir, train[0][0]))
    im.rotate(45).show()

def preprocess_image(x):
    x = x.astype(x, float32)
    x = np.array(x)
    x = np.divide(x, 255.0) 
    x = np.subtract(x, 1.0) 
    x = np.multiply(x, 2.0) 
    return x

def unprocess_image(x):
    x = np.divide(x, 2.0)
    x = np.add(x, 1.0)
    x = np.multiply(x, 255.0)
    x = x.astype(x, uint8)
    x = Image.fromarray(x)
    return x

def get_train_batch(batch_size, horiz_flip_prob=.5, rotate_prob=.2):
    # Grabs a random batch from availiable training data
    # params:
    #   batch_size = the number of training examples to return
    #   horiz_flip_prob (default .5) = probability of doing a horizontal flip
    #   rotate_prob (default .1) = probability of rotating +/-25 degrees
    samples = [[],[]]
    while len(samples[0]) < batch_size:
        sample = train[random.randint(0, num_train-1)]
        image = Image.open(join(train_image_dir, sample[0])).resize((299, 299), Image.BILINEAR).convert(mode='RGB')
        if random.random() < horiz_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < rotate_prob:
            image = image.rotate(random.randint(-25, 25))

        image = preprocess_image(image)
        label = np.zeros((200))
        label[sample[1]] = 1

        samples[0].append(image)
        samples[1].append(label)
    return samples

def get_val(batch_size=-1):
    # Grabs a  batch from availiable validation data
    # params:
    #   batch_size = the number of training examples to return, -1 for all
    validx = 0
    samples = [[],[]]
    while len(samples[0]) < batch_size:
        if batch_size == -1:
            sample = val[validx]
            validx += 1
        else:
            sample = val[random.randint(0, num_val-1)]

        image = preprocess_image(Image.open(
                join(train_image_dir, sample[0])
            ).resize((299, 299), Image.BILINEAR).convert(mode='RGB'))
        label = np.zeros((200))
        label[sample[1]] = 1

        samples[0].append(image)
        samples[1].append(label)
    return samples
