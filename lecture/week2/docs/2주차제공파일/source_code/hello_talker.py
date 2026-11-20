#!/usr/bin/env python3
"""Publish a greeting periodically."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloTalker(Node):
    """Publish an incrementing Hello ROS 2 message."""

    def __init__(self) -> None:
        super().__init__("hello_talker")
        
        # TODO: create_publisher()를 사용하여 String 메시지 타입으로 "/hello_topic" 토픽을 생성하시오. (큐 크기: 10)
        self._publisher = 
        
        self._count = 0
        
        # TODO: create_timer()를 사용하여 1.0초 주기마다 self._publish_greeting 콜백 함수를 호출하도록 타이머를 설정
        self._timer = 
        
        self.get_logger().info("Publishing greetings on /hello_topic")

    def _publish_greeting(self) -> None:
        self._count += 1
        
        message = String()
        message.data = f"Hello ROS2 #{self._count}"
        # TODO: _publisher로 message 발행
        self._publisher.
        
        self.get_logger().info(f'Published: "{message.data}"')


def main(args=None) -> None:
    """Run the talker node."""
    rclpy.init(args=args)
    node = HelloTalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
