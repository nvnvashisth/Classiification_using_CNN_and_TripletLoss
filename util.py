
from PIL import Image
import tensorflow as tf
import math
from decimal import Decimal


import os
import numpy as np
import matplotlib.pyplot as plt

class Sample:
    def __init__(self):
        self.img = []
        self.pose = []

    def initial(self, img, pose):
        self.img = img
        self.pose = pose


##Utility function for data preparation started ###################################
def getSubDir(path):
    return next(os.walk(path))[1]

def assertDimsCheck(set):
    print(set["img"].shape, set["pose"].shape,set["label"].shape)


def read_indicies(file_path,file_name):

    num_class = 5
    img_per_class = 1178
    train_idx = []
    f = open(os.path.join(file_path, file_name), 'r')
    line = f.readline()
    indices = line.split(',')
    indices = list(map(int, indices))
    f.close()
    # print(indices)
    # print(int(indices[-1]))
    for i in range(num_class):
        for j in indices:
            #print(i*img_per_class + j)
            train_idx.append(i*img_per_class + j)

    test_idx = [i for i in range(img_per_class * num_class) if i not in train_idx]

    return train_idx,test_idx

def loadDataset(dataSetPath,poseFilePath):

    # define the data container
    quateronins_list = []
    img_list = []
    class_list = []
    dataset = {}

    classes = getSubDir(dataSetPath)
    print (classes)
    # print(classes[0])
    p = classes[3]

    # for filename in os.listdir(curr_dir):
    # if filename.endswith(".png"):
    # img_file = (os.path.join(curr_dir, filename))
    # print(img_file)
    # img  = plt.imread(img_file)
    # continue

    #if(p):
    class_idx = 0
    for p in sorted(classes):
        #print(p)
        curr_dir = os.path.join(dataSetPath,p)
        pose_file = open(os.path.join(curr_dir,poseFilePath),'r')

        line = pose_file.readline()
        while line:
            if line.strip('\n').endswith(".png"):
                #print(line)
                img_file_name = line.split(" ")[1].strip('\n')
                #print(line.split(" ")[1].strip('\n'))
                img = plt.imread(os.path.join(curr_dir, img_file_name))
                img_list.append(img)
                #print(img.shape)

                line = pose_file.readline()
                qt_pose = line.split(" ")
                qt_pose[3] = qt_pose[3].strip('\n')
                quateronins_list.append(qt_pose)

                #class_list.append(p)
                class_list.append(class_idx)
                # print (qt_pose[3].strip('\n'))
            line = pose_file.readline()

        # bind the data into dicts and return the dict
        dataset['img'] = np.asarray(img_list)
        dataset['pose'] = np.asarray(quateronins_list, dtype= np.float32)
        dataset['label'] = np.asarray(class_list)
        class_idx = class_idx + 1

    assertDimsCheck(dataset)
    return dataset

def build_train_set(real_set,fine_set,train_idx):
    print('building train set')
    #train_idx,test_idx = util.read_indicies(os.path.join(base_path, "dataset\\real"), 'training_split.txt')

    #definig containers
    train_dict = {}

    #adding real image
    img_train_dict = real_set['img'][train_idx[:], :, :, :]
    pose_train_dict = real_set['pose'][train_idx[:], :]
    label_train_dict = real_set['label'][train_idx[:]]

    # print(img_train_dict.shape)
    # print(pose_train_dict.shape)
    # print(label_train_dict.shape)
    #util.assertDimsCheck(train_set_dict)

    # adding the synthetic image
    # adding fine set to the test set
    processed_train = np.concatenate((img_train_dict,fine_set['img']), axis= 0)
    # processed_train = (processed_train - np.mean(processed_train, axis = 0))
    # processed_train = processed_train / np.std(processed_train,axis = 0)
    # print(processed_train.shape)
    train_dict['img'] = processed_train
    train_dict['pose'] = np.concatenate((pose_train_dict,fine_set['pose']), axis= 0)
    train_dict['label'] = np.concatenate((label_train_dict,fine_set['label']), axis= 0)

    assertDimsCheck(train_dict)
    return train_dict


def build_test_set(real_set,test_idx):
    print('building test set')
    #train_idx, test_idx = util.read_indicies(os.path.join(base_path, "dataset\\real"), 'training_split.txt')

    test_dict = {}
    test_list = real_set['img'][test_idx[:], :, :, :]
    processed_test = (test_list - np.mean(test_list, axis=0))
    processed_test = processed_test / np.std(processed_test, axis=0)
    #print(processed_test.shape)

    test_dict['img'] = processed_test
    test_dict['pose'] = real_set['pose'][test_idx[:], :]
    test_dict['label'] = real_set['label'][test_idx[:]]

    assertDimsCheck(test_dict)
    return test_dict

##Utility function for data preparation started ###################################


##Utility function for data preparation started ###################################




def normalize_img(img):
    
    for i in range(3): #3 channels between [0,1]
        min_val = np.min(img[:,:,i])
        max_val = np.max(img[:,:,i])

        img[:,:,i] -= min_val
        img[:,:,i] /= max_val

    return img

    
def batch_generator(train_img, train_class, train_quat, db_img, db_class, db_quat, batch_size):
    
    mask_random = np.random.choice(len(train_img), batch_size)

    #get anchors
    anchors_img = train_img[mask_random]
    anchors_class = train_class[mask_random]
    anchors_quat = train_quat[mask_random]
    
    #find puller
    puller_ind = []
    for i in range(0, batch_size):
        sel_indices = np.where(db_class == anchors_class[i])[0]

        min_angle = 181
        min_index = -1
        for j in range(0, len(sel_indices)):
            angle = quaternion_similarity(anchors_quat[i], db_quat[sel_indices[j]])

            if angle < min_angle:
                min_angle = angle
                min_index = sel_indices[j]
        puller_ind.append(min_index)            
            
    puller_img = db_img[puller_ind]
    puller_class = db_class[puller_ind]
    puller_quat = db_quat[puller_ind]

    #find pushers
    pusher_ind = []
    i = 0
    while len(pusher_ind) != batch_size:
        index = np.random.choice(len(db_img), 1)[0]
        
        if anchors_class[i] != db_class[index] or (anchors_class[i] == db_class[index] and quaternion_similarity(anchors_quat[i], db_quat[index]) > 91):
            pusher_ind.append(index)
            i += 1
    
    pusher_img = db_img[pusher_ind]
    pusher_class = db_class[pusher_ind]
    pusher_quat = db_quat[pusher_ind]
    
    #COMBINE ALL
    batch_img, batch_class, batch_quat = [], [], []
    for i in range(0, batch_size):
        batch_img.extend([anchors_img[i], puller_img[i], pusher_img[i]])
        batch_class.extend([anchors_class[i], puller_class[i], pusher_class[i]])
        batch_quat.extend([anchors_quat[i], puller_quat[i], pusher_quat[i]])
    
    return np.array(batch_img), np.array(batch_class), np.array(batch_quat)
    

def quaternion_similarity(q1, q2):
    # θ = arccos[ 2*⟨q1,q2⟩^2 − 1 ]
    return np.rad2deg(math.acos(2 * (np.clip((np.dot(q1, q2)), -1, +1) ** 2) - 1))
    
    
def plot_images(images):
    assert len(images) == 9
    
    labels = ["Anchor", "Puller", "Pusher"]
    
    # Figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')            
        ax.set_xlabel(labels[i % 3])
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

    
##################TENSORFLOW######################

def new_weights(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)


def new_biases(length, name=None):
    return tf.Variable(tf.constant(0.05, shape=[length]), name=name)


def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling=True, name=None):  
    
    shape = [filter_size, filter_size, num_input_channels, num_filters] #as defined in tensorflow documentation
    
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding="VALID", name=name)
    layer += biases  # Add biases to each filter-channel

    layer = tf.nn.relu(layer)

    if pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    return layer


def flatten_layer(layer, name=None):
    layer_shape = layer.get_shape()  # [num_images, img_height, img_width, num_channels]

    num_features = layer_shape[1:4].num_elements() # The number of features is: img_height * img_width * num_channels
    
    layer_flat = tf.reshape(layer, [-1, num_features], name=name)  # [num_images, img_height * img_width * num_channels]

    return layer_flat, num_features

#Fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, relu=True, name=None): 

    weights = new_weights(shape=[num_inputs, num_outputs], name=name)
    biases = new_biases(length=num_outputs, name=name)

    layer = tf.matmul(input, weights) + biases

    if relu:
        layer = tf.nn.relu(layer)

    return layer

#Calculation of total loss (Pair + triplets)
def total_loss(features, m = 0.01):
    return triplets_loss(features, m) + pairs_loss(features)
    
# Calculation of triplets loss    
def triplets_loss(feature_desc, m = 0.01):
    batch_size = feature_desc.shape[0]
    
    diff_pos = tf.reduce_sum(tf.square(feature_desc[0:batch_size:3] - feature_desc[1:batch_size:3]), 1)
    diff_neg = tf.reduce_sum(tf.square(feature_desc[0:batch_size:3] - feature_desc[2:batch_size:3]), 1)

    loss = tf.maximum(0., 1 - (diff_neg / (diff_pos + m)))
    loss = tf.reduce_sum(loss)
    
    return loss

#Calculation of pair loss
def pairs_loss(feature_desc):
    batch_size = feature_desc.shape[0]

    loss = tf.reduce_sum(tf.square(feature_desc[0:batch_size:3] - feature_desc[1:batch_size:3]), 1)
    loss = tf.reduce_sum(loss)
    
    return loss

# Output the descriptor from the database and test from each model saved at 1000's iteration

def output_features(S_db, S_test, output_layer, loss, x, index_model):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "./checkpoint/model" + str(index_model) +".ckpt") 
        #saver.restore(sess, "/checkpoint/model55000.ckpt")
        descriptors_db = sess.run(output_layer, feed_dict={x: S_db})
        tmp_1 = sess.run(output_layer, feed_dict={x: S_test[:1200]})
        tmp_2 = sess.run(output_layer, feed_dict={x: S_test[1200:2400]})
        tmp_3 = sess.run(output_layer, feed_dict={x: S_test[2400:]})

    descriptors_test = np.concatenate((tmp_1, tmp_2, tmp_3), axis = 0)    
    
    assert(len(descriptors_test) == len(S_test))
    assert(len(S_db) == len(descriptors_db))
    
    return descriptors_db, descriptors_test
        
# Calculating the angles count, indices and confusion matrix at each 1000's iteration.
# Calculating the nearest neighbour for the descriptor in db, if found, then calculating the angle difference and storing it
# Based on the storing value of angle, we are calculating the histograms
    
def matching_feature_map(feat_db, feat_test, db_classes, test_classes, db_quat, test_quat):
    indeces = []
    angles = [10.0, 20.0, 40.0, 180.0]
    angles_count = [0.0, 0.0, 0.0, 0.0]
    confusion_matrix = np.zeros((5,5))
    
    for i in range(len(feat_test)):

        min_dist = 1000000000.
        chosen_index = -1
        for j in range(len(feat_db)):
            dist = Euclidean_distance(feat_test[i], feat_db[j])

            if dist < min_dist:
                min_dist = dist
                chosen_index = j
        
        indeces.append(chosen_index)
        confusion_matrix[test_classes[i], db_classes[chosen_index]] += 1
        
        if db_classes[chosen_index] == test_classes[i]:
            quat_simil = quaternion_similarity(db_quat[chosen_index], test_quat[i])

            for k in range(len(angles)):
                if quat_simil <= angles[k]:
                    angles_count[k] += 1
    
    angles_count = np.array(angles_count)
    angles_count = (angles_count * 100.) / float(len(feat_test))
    
    return angles_count, indeces, confusion_matrix

# Calculate the euclidean distance

def Euclidean_distance(f1, f2):
    return np.sqrt(np.sum(np.square(f1 - f2)))

# Save the histogram at each 1000's iteration

def save_histogram(hist, index):
    assert(len(hist) == 4)
    
    angles = np.array(list(range(4)))
    hist_plot = plt.bar(angles, hist)
    
    plt.xticks(angles, ("10°", "20°", "40°", "180°"))

    plt.title('Iteration ' + str(index))
    plt.xlabel('Angle')
    plt.ylabel('Accuracy (%)')
    
    for rect in hist_plot:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % height, ha='center', va='bottom')
    
    plt.savefig('histograms/' + str(index) + '.png')
    plt.clf()
    plt.cla()
    plt.close()
    

# This function draws curve for angle < 10,20,40 and 180 for the final angles    

def save_plot_angles(histogram):
    
    histogram = np.array(histogram)

    indeces = np.array(list(range(0, len(histogram))))
    indeces *= 1000

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.plot(indeces, histogram[:,0], 'r-')
    ax1.set_title("Angle < 10°")

    ax2 = fig.add_subplot(222)
    ax2.plot(indeces, histogram[:,1], 'k-')
    ax2.set_title("Angle < 20°")

    ax3 = fig.add_subplot(223)
    ax3.plot(indeces, histogram[:,2], 'b-')
    ax3.set_title("Angle < 40°")

    ax4 = fig.add_subplot(224)
    ax4.plot(indeces, histogram[:,3], 'g-')
    ax4.set_title("Angle < 180°")

    plt.tight_layout()
    fig = plt.gcf()

    plt.savefig('histograms/final_angles.png')    
   
    plt.clf()
    plt.cla()
    plt.close()


