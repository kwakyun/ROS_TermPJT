#!/usr/bin/env python3
"""Subscribe to greetings and print them."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloListener(Node):
    """Print every message received from the talker."""

    def __init__(self) -> None:
        super().__init__("hello_listener")

        # create_subscription()을 사용하여 String 메시지 타입으로 "/hello_topic" 토픽 구독, self._on_message 콜백 함수 연결 (큐 크기: 10)
        self._subscription = self.create_subscription(
            String, "/hello_topic", self._on_message, 10
        )
        self.get_logger().info("Waiting for greetings on /hello_topic")

    def _on_message(self, message: String) -> None:
        self.get_logger().info(f'Received: "{message.data}"')


def main(args=None) -> None:
    """Run the listener node."""
    rclpy.init(args=args)
    node = HelloListener()
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
