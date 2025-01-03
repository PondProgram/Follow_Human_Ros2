#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import csv
import math
import numpy as np

class RobotWaypointFollower(Node):
    def __init__(self):
        super().__init__('robot_waypoint_follower')
        
        # สร้าง publisher สำหรับส่งคำสั่งความเร็ว
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # สร้าง subscriber สำหรับรับข้อมูลตำแหน่งปัจจุบัน
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        
        # ตัวแปรสำหรับเก็บตำแหน่งปัจจุบัน
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # ค่าพารามิเตอร์สำหรับการควบคุม
        self.linear_speed = 0.2
        self.angular_speed = 0.3
        self.position_tolerance = 0.15
        self.yaw_tolerance = 0.15
        
        self.rotation_start_threshold = 0.2
        self.rotation_stop_threshold = 0.1
        
        self.prev_angular_vel = 0.0
        self.angular_vel_smoothing = 0.3
        
        # อ่านเส้นทางจากไฟล์ CSV
        self.waypoints = self.load_waypoints('/home/thanawat/amr_ws/robot_poses.csv')
        self.waypoints.reverse()
        self.current_waypoint_index = 0
        
        # สร้าง timer สำหรับ control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('เริ่มต้นการเคลื่อนที่ย้อนกลับ')

    def quaternion_to_yaw(self, x, y, z, w):
        """
        แปลง quaternion เป็นมุม yaw โดยไม่ใช้ euler_from_quaternion
        """
        # คำนวณ yaw จาก quaternion โดยตรง
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                waypoint = {
                    'x': float(row['X']),
                    'y': float(row['Y']),
                    'yaw': float(row['Yaw(rad)'])
                }
                waypoints.append(waypoint)
        return waypoints

    def odom_callback(self, msg):
        # อัพเดตตำแหน่งปัจจุบัน
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # แปลง quaternion เป็น yaw โดยใช้ฟังก์ชันที่เขียนขึ้นเอง
        orientation_q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        )

    def get_distance_to_goal(self, goal_x, goal_y):
        return math.sqrt((goal_x - self.current_x)**2 + (goal_y - self.current_y)**2)

    def get_angle_to_goal(self, goal_x, goal_y):
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        target_angle = math.atan2(dy, dx)
        
        angle_diff = target_angle - self.current_yaw
        
        # ปรับมุมให้อยู่ในช่วง -pi ถึง pi
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
                
        return angle_diff

    def control_loop(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info('ถึงจุดหมายแล้ว')
            return

        goal = self.waypoints[self.current_waypoint_index]
        distance = self.get_distance_to_goal(goal['x'], goal['y'])
        angle = self.get_angle_to_goal(goal['x'], goal['y'])

        cmd = Twist()
        
        if distance <= self.position_tolerance:
            self.current_waypoint_index += 1
            self.prev_angular_vel = 0.0
            self.get_logger().info(f'ถึงจุดที่ {self.current_waypoint_index}/{len(self.waypoints)}')
        else:
            if abs(angle) > self.rotation_start_threshold:
                target_angular_vel = self.angular_speed * math.copysign(1, angle) * min(1.0, abs(angle))
                cmd.angular.z = (self.prev_angular_vel * (1 - self.angular_vel_smoothing) + 
                            target_angular_vel * self.angular_vel_smoothing)
                self.prev_angular_vel = cmd.angular.z
                self.get_logger().info(f'กำลังหมุน: {math.degrees(angle):.2f}°')
            else:
                cmd.linear.x = min(self.linear_speed, distance)
                cmd.angular.z = angle * 0.5
                self.get_logger().info(f'กำลังเคลื่อนที่: ระยะทาง={distance:.2f}m, มุม={math.degrees(angle):.2f}°')

        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

def main():
    rclpy.init()
    node = RobotWaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
