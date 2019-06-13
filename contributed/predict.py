from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ----------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Rishi Rai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------
import cv2
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle

from facenet.src.align import detect_face
from facenet.src.facenet import prewhiten, load_model
from sklearn.svm import SVC
from scipy import misc
from six.moves import xrange

import imageio


def main(args):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)

            # load Video
            capture = cv2.VideoCapture('../pretrained/AI_IU_02.mp4')
            _, frame = capture.read()
            sx, sy, sz = frame.shape
            scaler = 1

            while True:
                ret, frame = capture.read()
                if not ret: break
                frame = cv2.resize(frame, (int(sy * scaler), int(sx * scaler)))
                ret, images, recs, cout_per_image, nrof_samples = load_and_align_data(frame, args.image_size,
                                                                                      args.margin,
                                                                                      args.gpu_memory_fraction,
                                                                                      (pnet, rnet, onet))
                if not ret: continue
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)

                predictions = model.predict_proba(emb)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                k = 0
                # print predictions
                for i in range(nrof_samples):
                    print("\npeople in image %s :" % (args.image_files[i]))
                    for j in range(cout_per_image[i]):
                        if best_class_probabilities[k] < 0.5:
                            continue
                        cv2.putText(frame, class_names[best_class_indices[k]], (int(recs[j][0]), int(recs[j][1] - 4)),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, pt1=(recs[j][0], recs[j][1]), pt2=(recs[j][2], recs[j][3]),
                                      color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                        k += 1

                cv2.imshow('frame', frame)
                cv2.waitKey(1)


def load_and_align_data(img, image_size, margin, gpu_memory_fraction, nets):
    ret = True
    images = []
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    pnet, rnet, onet = nets

    nrof_samples = 1
    img_list = []
    count_per_image = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    count_per_image.append(len(bounding_boxes))
    recs = []

    for j in range(len(bounding_boxes)):
        det = np.squeeze(bounding_boxes[j, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        recs.append(bb)
        aligned = cv2.resize(cropped, (image_size, image_size))
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    if len(img_list):
        images = np.stack(img_list)
    else:
        ret = False
    return ret, images, recs, count_per_image, nrof_samples


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_files', type=str, nargs='+', help='Path(s) of the image(s)',
                        default='pretrained/train/sana/SANA_6.jpg')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='../pretrained/20180402-114759/20180402-114759.pb')
    parser.add_argument('--classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.',
                        default='../pretrained/my_classifier.pkl')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
