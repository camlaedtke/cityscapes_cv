from __future__ import print_function, absolute_import, division
from collections import namedtuple
import numpy as np
import tensorflow as tf
from matplotlib import colors
import matplotlib.pyplot as plt


def parse_record(raw_record):
    keys_to_features = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string),
      'image/format': tf.io.FixedLenFeature((), tf.string),
      'image/height': tf.io.FixedLenFeature((), tf.int64),
      'image/width': tf.io.FixedLenFeature((), tf.int64),
      'image/channels': tf.io.FixedLenFeature((), tf.int64),
      'label/encoded': tf.io.FixedLenFeature((), tf.string),
      'label/format': tf.io.FixedLenFeature((), tf.string),
    }

    parsed = tf.io.parse_single_example(raw_record, keys_to_features)

    image = tf.image.decode_image(tf.reshape(parsed['image/encoded'], shape=[]), 3)
    image = tf.cast(image, tf.float32)
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.cast(label, tf.int32)
    label.set_shape([None, None, 1])

    return image, label




def get_dataset_from_tfrecord(tfrecord_dir):
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_dir)
    parsed_dataset = tfrecord_dataset.map(parse_record)
    return parsed_dataset


def get_labels():

    # a label and all meta information
    Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).
                        # Do not modify these IDs, since exactly these IDs are expected by the
                        # evaluation server.

        'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                        # ground truth images with train IDs, using the tools provided in the
                        # 'preparation' folder. However, make sure to validate or submit results
                        # to our evaluation server using the regular IDs above!
                        # For trainIds, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.
                        # Max value is 255!

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )


    #--------------------------------------------------------------------------------
    # A list of all labels
    #--------------------------------------------------------------------------------

    # Please adapt the train IDs as appropriate for your approach.
    # Note that you might want to ignore labels with ID 255 during training.
    # Further note that the current train IDs are only a suggestion. You can use whatever you like.
    # Make sure to provide your results using the original IDs and not the training IDs.
    # Note that many IDs are ignored in evaluation and thus you never need to predict these!
    

    labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,        0 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,        0 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        0 , 'flat'            , 0       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        0 , 'flat'            , 0       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        0 , 'construction'    , 0       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        0 , 'construction'    , 0       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        0 , 'construction'    , 0       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,        0 , 'object'          , 0       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        0 , 'vehicle'         , 0       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        0 , 'vehicle'         , 0       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 0       , False        , True         , (  0,  0,142) ),
    ]

    return labels

