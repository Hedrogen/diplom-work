#!/usr/bin/env python3

import cv2
import numpy as np
from time import sleep

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoTalker(Node):
    def __init__(self):
        print('Start!')
        super().__init__('video_talker')
        self.publisher_ = self.create_publisher(Image, 'video_topic', 10)
        self.timer = self.create_timer(0.03, self.timer_callback)

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture('video.mov')
        # self.cap = cv2.VideoCapture('video_example.mp4')
        self.ret, self.frame = self.cap.read()
        self.i = 0

    def timer_callback(self):
        self.get_logger().info(f'Publish_image: #{self.i}')
        self.ret, self.frame = self.cap.read()
        if self.ret:
            img_msg = self.bridge.cv2_to_imgmsg(self.frame, "bgr8")
            self.publisher_.publish(img_msg)
            sleep(0.03)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    video_talker = VideoTalker()
    rclpy.spin(video_talker)
    video_talker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
