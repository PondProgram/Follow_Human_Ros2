#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
from datetime import datetime
import math
from tf_transformations import euler_from_quaternion

class RobotPoseLogger(Node):
    def __init__(self):
        super().__init__('robot_pose_logger')
        
        # สร้างไฟล์ CSV และเขียนส่วนหัว
        self.csv_filename = f'robot_poses.csv'
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'X', 'Y', 'Z', 'Yaw(rad)', 'Yaw(deg)'])
        
        # สร้าง subscriber สำหรับ odometry
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',  # topic name
            self.odom_callback,
            10)
        
        # เก็บตำแหน่งล่าสุดที่บันทึก
        self.last_saved_x = 0.0
        self.last_saved_y = 0.0
        self.is_first_point = True  # ตัวแปรสำหรับจุดแรก
        
    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
    def odom_callback(self, msg: Odometry):
        try:
            # ดึงค่าพิกัดจาก odometry
            current_x = msg.pose.pose.position.x
            current_y = msg.pose.pose.position.y
            current_z = msg.pose.pose.position.z
            
            # ถ้าเป็นจุดแรก หรือ ระยะทางห่างจากจุดล่าสุดมากกว่า 1 เมตร จึงบันทึก
            distance = self.calculate_distance(current_x, current_y, 
                                            self.last_saved_x, self.last_saved_y)
            
            if self.is_first_point or distance >= 1.0:
                # ดึงค่า quaternion
                orientation_q = msg.pose.pose.orientation
                q = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                
                # แปลง quaternion เป็น euler angles (roll, pitch, yaw)
                roll, pitch, yaw = euler_from_quaternion(q)
                
                # แปลง yaw เป็นองศา
                yaw_deg = math.degrees(yaw)
                
                # บันทึกลงไฟล์ CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([timestamp, current_x, current_y, current_z, yaw, yaw_deg])
                
                # อัพเดทตำแหน่งล่าสุดที่บันทึก
                self.last_saved_x = current_x
                self.last_saved_y = current_y
                self.is_first_point = False  # ไม่ใช่จุดแรกแล้ว
                
                self.get_logger().info(f'Save: X={current_x:.2f}, Y={current_y:.2f}, Z={current_z:.2f}, '
                                     f'Yaw={yaw_deg:.2f}°, Distance={distance:.2f}m')
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')

def main():
    rclpy.init()
    node = RobotPoseLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()