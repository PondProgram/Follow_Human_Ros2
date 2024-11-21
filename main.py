import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2

thres = 0.6
nmsThres = 0.2
cap = cv2.VideoCapture(2) 
cap.set(3, 640)
cap.set(4, 480)

# Load COCO class names
classNames = []
classFile = '/home/thanawat/amr_ws/src/follow_human/follow_human/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().strip().split('\n')

# Load model configuration and weights
configPath = '/home/thanawat/amr_ws/src/follow_human/follow_human/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/thanawat/amr_ws/src/follow_human/follow_human/frozen_inference_graph.pb'

# Load the pre-trained model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.twist = Twist()
        self.threshold = 80


    def move_robot(self, x_deviation, y_deviation):
        max_linear_speed = 0.1  
        max_angular_speed = 0.1  

        twist = Twist()

        if x_deviation is not None and y_deviation is not None:
            if y_deviation > 20:
                linear_speed = max_linear_speed * (y_deviation / (y_deviation + self.threshold))
                twist.linear.x = min(linear_speed, max_linear_speed)
                print(f"......................Move Forward with speed: {twist.linear.x:.2f}......................")
            else:
                twist.linear.x = 0.0
                print("......................Robot Stop......................")

            if abs(x_deviation) > self.threshold:
                angular_speed = max_angular_speed * (x_deviation / (abs(x_deviation) + self.threshold))
                twist.angular.z = max(min(angular_speed, max_angular_speed), -max_angular_speed)
                if x_deviation > 0:
                    print(f"......................Turn Left with speed: {twist.angular.z:.2f}......................")
                else:
                    print(f"......................Turn Right with speed: {twist.angular.z:.2f}......................")
            else:
                twist.angular.z = 0.0

        self.publisher_.publish(twist)

def main():
    rclpy.init()
    robot_controller = RobotController()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera. Exiting...")
                break

            height, width, _ = frame.shape
            classIds, confs, bbox = net.detect(frame, confThreshold=thres, nmsThreshold=nmsThres)
            center_x = width // 2
            center_y = height // 2
            cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 255), 2)
            cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)
            cv2.line(frame, (center_x - robot_controller.threshold, 0), 
                     (center_x - robot_controller.threshold, height), (0, 0, 255), 2)
            cv2.line(frame, (center_x + robot_controller.threshold, 0), 
                     (center_x + robot_controller.threshold, height), (0, 0, 255), 2)

            x_deviation = None
            if len(classIds) > 0:
                for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if classId == 1:  # Detect only 'person'
                        x, y, w, h = box
                        center_x_obj = x + w // 2
                        center_y_obj = y + h // 2
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x_obj, center_y_obj), 5, (255, 0, 0), -1)
                        x_deviation = round((width // 2) - center_x_obj, 3)
                        y_deviation = round((height - (y + h)), 3)

                        # Move the robot based on x_deviation
                        if x_deviation is not None and y_deviation is not None:
                            robot_controller.move_robot(x_deviation, y_deviation)

            cv2.imshow("Image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
