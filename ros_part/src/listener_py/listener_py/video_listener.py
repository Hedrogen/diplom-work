#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import tensorflow as tf
import numpy as np
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


PB_PATH = 'machine_view/model'
LABELS_PATH = 'machine_view/label_map.pbtxt'
CATEGORY_INDEX = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)


def recognize_object(model, img):
    image = np.asarray(img)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


def process_image(model, img):
    output_dict = recognize_object(model, img)
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        CATEGORY_INDEX,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    return output_dict, img


class VideoListener(Node):
    def __init__(self):
        super().__init__('video_listener')
        self.subscription = self.create_subscription(Image, 'video_topic', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.tf_model = tf.saved_model.load(PB_PATH)

    def listener_callback(self, msg):
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, (600, 600))
        res_info, result_img = process_image(self.tf_model, frame)
        cv2.imshow('Video with text', result_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    video_listener = VideoListener()
    rclpy.spin(video_listener)
    video_listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
