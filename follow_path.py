#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import csv
import math
from tf_transformations import euler_from_quaternion
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
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.3  # ลดความเร็วการหมุนลง
        self.position_tolerance = 0.15  # เพิ่มค่า tolerance ระยะทาง
        self.yaw_tolerance = 0.15  # เพิ่มค่า tolerance มุม
        
        # เพิ่มค่า hysteresis
        self.rotation_start_threshold = 0.2  # เริ่มหมุนเมื่อเบี่ยงเบนมากกว่านี้
        self.rotation_stop_threshold = 0.1   # หยุดหมุนเมื่อเบี่ยงเบนน้อยกว่านี้
        
        # เพิ่มตัวแปรสำหรับ smoothing
        self.prev_angular_vel = 0.0
        self.angular_vel_smoothing = 0.3  # ค่าการ smooth ความเร็วเชิงมุม
        
        # อ่านเส้นทางจากไฟล์ CSV
        self.waypoints = self.load_waypoints('/home/thanawat/amr_ws/robot_poses.csv')
        self.waypoints.reverse()  # กลับลำดับจุดเพื่อย้อนกลับ
        self.current_waypoint_index = 0
        
        # สร้าง timer สำหรับ control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('เริ่มต้นการเคลื่อนที่ย้อนกลับ')

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
        
        # แปลง quaternion เป็น yaw
        orientation_q = msg.pose.pose.orientation
        q = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_yaw = euler_from_quaternion(q)

    def get_distance_to_goal(self, goal_x, goal_y):
        return math.sqrt((goal_x - self.current_x)**2 + (goal_y - self.current_y)**2)

    def get_angle_to_goal(self, goal_x, goal_y):
        # คำนวณมุมที่ต้องหันไปยังเป้าหมาย
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        target_angle = math.atan2(dy, dx)
        
        # คำนวณความต่างของมุม
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
        
        # ตรวจสอบว่าถึงจุดหมายหรือยัง
        if distance <= self.position_tolerance:
            # เมื่อถึงจุดหมาย ไปยังจุดถัดไปได้เลย โดยไม่ต้องปรับมุมสุดท้าย
            self.current_waypoint_index += 1
            self.prev_angular_vel = 0.0  # รีเซ็ตค่า smoothing
            self.get_logger().info(f'ถึงจุดที่ {self.current_waypoint_index}/{len(self.waypoints)}')
        else:
            # ยังไม่ถึงจุดหมาย ต้องเคลื่อนที่ต่อ
            if abs(angle) > self.rotation_start_threshold:
                # หมุนเพื่อปรับทิศทาง
                target_angular_vel = self.angular_speed * math.copysign(1, angle) * min(1.0, abs(angle))
                cmd.angular.z = (self.prev_angular_vel * (1 - self.angular_vel_smoothing) + 
                            target_angular_vel * self.angular_vel_smoothing)
                self.prev_angular_vel = cmd.angular.z
                self.get_logger().info(f'กำลังหมุน: {math.degrees(angle):.2f}°')
            else:
                # เคลื่อนที่ไปข้างหน้า
                cmd.linear.x = min(self.linear_speed, distance)
                # ปรับมุมเล็กน้อยขณะเคลื่อนที่
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