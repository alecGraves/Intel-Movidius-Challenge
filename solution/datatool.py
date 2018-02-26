##########################################################
# A generic datatool for loading jpegs and csv labels
#   built for the Intel Movidius Competition
# copyright shadySource
# MIT Liscense
##########################################################
import random
from os.path import join
import cv2
import numpy as np

random.seed(13037)


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

num_val = 256
val = labels[-num_val:]
assert num_val == len(val)

num_train = len(labels) - num_val
train = labels[:num_train]
assert num_train == len(train)

if debug:
    print('There are', len(labels), 'labels.')
    print('Reserving', num_train, 'labels for training')
    print('Reserving', num_val, 'labels for testing')

# explore number of each category(they are all 400 samples)
# label_baskets = [0]*200
# for label in labels:
#     label_baskets[label[1]-1] += 1
# print(max(label_baskets))
# print(min(label_baskets))

# if debug: # removed PIL, will no longer work.
#     im = Image.open(join(train_image_dir, train[0][0]))
#     im.rotate(45).show()

def preprocess_image(x):
    x = np.array(x).astype(np.float32)
    x = np.array(x)
    x = np.divide(x, 255.0) 
    x = np.subtract(x, 0.5) 
    x = np.multiply(x, 2.0) 
    return x

def unprocess_image(x):
    x = np.divide(x, 2.0)
    x = np.add(x, 0.5)
    x = np.multiply(x, 255.0)
    x = x.astype(np.uint8)
    return x

TRAIN_IDX = 0
def get_train_batch(batch_size, horiz_flip_prob=.5, cutout=True):
    # Grabs a random batch from availiable training data
    # params:
    #   batch_size = the number of training examples to return
    #   horiz_flip_prob (default .5) = probability of doing a horizontal flip
    global TRAIN_IDX
    samples = [[],[]]
    while len(samples[0]) < batch_size:
        sample = train[TRAIN_IDX]
        TRAIN_IDX = (TRAIN_IDX + 1) % num_train
        image = cv2.imread(join(train_image_dir, sample[0]))
        image = cv2.resize(image, (299,299))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if random.random() < horiz_flip_prob:
            image = cv2.flip(image, 1)

        if cutout:
            cutout_scale = .4
            cutsize = [int(image.shape[0]*cutout_scale), int(image.shape[1]*cutout_scale)]
            x_offset = random.randint(0, image.shape[0]-cutsize[0])
            y_offset = random.randint(0, image.shape[1]-cutsize[1])
            image[x_offset:cutsize[0], y_offset:cutsize[1], :] = [127, 127, 127]

        image = preprocess_image(image)
        label = np.zeros((200))
        label[sample[1]] = 1

        samples[0].append(image)
        samples[1].append(label)
    return (np.array(samples[0]), np.array(samples[1]))

validx = 0
def get_val(batch_size):
    # Grabs a  batch from availiable validation data
    # params:
    global validx
    samples = [[],[]]
    while len(samples[0]) < batch_size:
        sample = val[validx]
        validx = (validx + 1) % num_val
                
        image = cv2.imread(join(train_image_dir, sample[0]))
        image = cv2.resize(image, (299,299))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)

        label = np.zeros((200))
        label[sample[1]] = 1

        samples[0].append(image)
        samples[1].append(label)
    return (np.array(samples[0]), np.array(samples[1]))
