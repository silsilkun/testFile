"""
🏗️ Smart Tower Builder - Main
==============================
기존 팀원 코드 구조 기반 + 새 비전 시스템
"""

import cv2
import rclpy
from rclpy.node import Node
import numpy as np

import DR_init
from tower_builder.gripper_drl_controller import GripperController
from tower_builder.camera import BlockDetectionSystem, Block

# ============================================================
# 로봇 설정 (반드시 클래스 정의 전에!)
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 200, 100

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


# ============================================================
# 메인 컨트롤러 (기존 구조 유지)
# ============================================================

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        self.get_logger().info("RobotControllerNode 시작")

        # =========================
        # ✅ 새 비전 시스템 사용
        # =========================
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            self.get_logger().error("BlockDetectionSystem start() 실패")
            raise RuntimeError("Vision system start failed")

        self.blocks = []
        self.selected_block = None

        # =========================
        # 로봇 및 그리퍼 초기화 (기존 유지)
        # =========================
        self.gripper = None
        try:
            from DSR_ROBOT2 import wait
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            self.get_logger().info("Waiting for service /dsr01/drl/drl_start...")
            wait(2)

            if not self.gripper.initialize():
                self.get_logger().error("Gripper initialization failed. Exiting.")
                raise Exception("Gripper initialization failed")

            self.get_logger().info("그리퍼를 활성화합니다...")
            self.gripper.move(0)
        except Exception as e:
            self.get_logger().error(f"An error occurred during gripper setup: {e}")
            raise

        self.get_logger().info("초기화 완료")

    def stop_camera(self):
        if self.vision:
            self.vision.stop()

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    # =========================
    # 매 프레임 처리
    # =========================
    def process_and_render(self):
        # 트랙바 값 반영
        self.vision.config.threshold = cv2.getTrackbarPos("Threshold", "Control")
        self.vision.config.min_area = cv2.getTrackbarPos("Min Area", "Control")
        self.vision.config.max_area = cv2.getTrackbarPos("Max Area", "Control")

        # 비전 업데이트
        if not self.vision.update():
            return

        self.blocks = self.vision.last_blocks
        color = self.vision.last_frame
        depth_img = self.vision.last_depth

        # ===== Result 창 =====
        display = color.copy()
        cfg = self.vision.config

        # ROI 박스
        cv2.rectangle(display, (cfg.roi_x, cfg.roi_y), 
                     (cfg.roi_x + cfg.roi_w, cfg.roi_y + cfg.roi_h), (0, 0, 255), 2)
        cv2.putText(display, "ROI (WHITE)", (cfg.roi_x, cfg.roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 블록 표시
        for block in self.blocks:
            is_sel = (block == self.selected_block)
            color_draw = (0, 255, 255) if is_sel else (0, 255, 0)
            thickness = 3 if is_sel else 2

            cv2.drawContours(display, [block.rotated_box], 0, color_draw, thickness)

            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

            if block.is_valid:
                txt_depth = f"{block.depth_cm:.0f}cm"
                cv2.putText(display, txt_depth, (cx - 15, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2)

                txt_size = f"{block.real_width_mm:.0f}x{block.real_height_mm:.0f}mm"
                cv2.putText(display, txt_size, (cx - 30, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.putText(display, "no depth", (cx - 25, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.putText(display, f"({cx},{cy})", (cx - 25, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(display, f"Blocks: {len(self.blocks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Result", display)

        # ===== Binary (ROI) 창 =====
        if self.vision.detector.binary_view is not None:
            cv2.imshow("Binary (ROI)", self.vision.detector.binary_view)

        # ===== Depth 창 =====
        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )

        valid_mask = depth_img > 0
        if np.any(valid_mask):
            scale = self.vision.camera.depth_scale
            min_d = np.min(depth_img[valid_mask]) * scale * 100
            max_d = np.max(depth_img[valid_mask]) * scale * 100
            cv2.putText(depth_display, f"Range: {min_d:.0f}-{max_d:.0f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Depth", depth_display)

    # =========================
    # ✅ 마우스 클릭: rotated_box 기반으로 블록 인식
    # =========================
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # ✅ 새 방식: rotated_box 기반 클릭 판정
        block = self.vision.find_block_at(x, y, update=False)
        
        if block:
            # 블록 클릭됨!
            self.selected_block = block
            cx, cy = block.center_2d
            
            print(f"\n🎯 블록 선택됨!")
            print(f"   클릭 위치: ({x}, {y}) → 블록 중심: ({cx}, {cy})")
            
            if not block.is_valid:
                print("   ⚠️ 깊이 정보 없음")
                return
            
            # 3D 좌표 (카메라 기준, mm)
            cam_x, cam_y, cam_z = block.center_3d_mm
            
            # 로봇 좌표로 변환 (기존 변환 로직)
            final_x = 685 + cam_y
            final_y = cam_x + 20
            final_z = 810 - cam_z
            
            if final_z <= 150:
                final_z = 150

            print(f"   카메라 좌표: X={cam_x:.1f}, Y={cam_y:.1f}, Z={cam_z:.1f} mm")
            print(f"   로봇 좌표: X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f} mm")

            self.move_robot_and_control_gripper(final_x, final_y, final_z)
            print("=" * 50)
        else:
            # 빈 공간 클릭
            self.selected_block = None
            depth = self.vision.get_depth_at(x, y)
            
            print(f"\n빈 공간 클릭 ({x}, {y})")
            if depth > 0:
                print(f"   → 해당 위치 뎁스: {depth*100:.1f}cm")
                point_3d = self.vision.get_3d_at(x, y)
                if point_3d:
                    print(f"   → 3D 좌표: X={point_3d[0]*1000:.1f}mm, "
                          f"Y={point_3d[1]*1000:.1f}mm, Z={point_3d[2]*1000:.1f}mm")
            else:
                print(f"   → 뎁스 측정 불가")

    def move_robot_and_control_gripper(self, x, y, z):
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj

        try:
            current_pos = get_current_posx()[0]
            cur_x, cur_y, cur_z, cur_Rx, cur_Ry, cur_Rz = current_pos

            approach_lift_mm = 50.0

            target_up = [x, y, z + approach_lift_mm, cur_Rx, cur_Ry, cur_Rz]
            target_at = [x, y, z, cur_Rx, cur_Ry, cur_Rz]
            home_pose = posj(0, 0, 90, 0, 90, 0)

            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            self.get_logger().info(f"→ 접근/하강 이동: {target_at}")
            movel(posx(target_at), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 그리퍼 닫기 (집기)
            if self.gripper:
                self.gripper.move(550)
            wait(1.5)

            # 다시 들어올리기
            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 드롭 지점
            drop_offset = 100.0
            drop_target_up = [x, y + drop_offset, z + approach_lift_mm, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(drop_target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            drop_target_down = [x, y + drop_offset, z, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(drop_target_down), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 그리퍼 열기 (놓기)
            if self.gripper:
                self.gripper.move(100)
            wait(1.0)

            movel(posx(drop_target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            self.get_logger().info("🏠 홈 자세로 복귀합니다.")
            movej(home_pose, VELOCITY, ACC)
            wait(1.0)

        except Exception as e:
            self.get_logger().error(f"로봇 이동 및 그리퍼 제어 중 오류 발생: {e}")


def main(args=None):
    rclpy.init(args=args)

    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except ImportError as e:
        print(f"DSR_ROBOT2 라이브러리를 임포트할 수 없습니다: {e}")
        rclpy.shutdown()
        return

    robot_controller = RobotControllerNode()

    # ===== 창/트랙바 구성 =====
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot_controller.mouse_callback)

    cv2.namedWindow("Control")
    cfg = robot_controller.vision.config
    cv2.createTrackbar("Threshold", "Control", cfg.threshold, 255, lambda x: None)
    cv2.createTrackbar("Min Area", "Control", cfg.min_area, 5000, lambda x: None)
    cv2.createTrackbar("Max Area", "Control", cfg.max_area, 30000, lambda x: None)

    print("\n" + "=" * 50)
    print("🏗️ Smart Tower Builder")
    print("=" * 50)
    print("📌 조작:")
    print("   - 블록 클릭: 블록 중심으로 이동 & 집기")
    print("   - 빈 공간 클릭: 해당 위치 깊이 확인")
    print("   - ESC: 종료")
    print("=" * 50 + "\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(robot_controller, timeout_sec=0.001)
            rclpy.spin_once(dsr_node, timeout_sec=0.001)

            robot_controller.process_and_render()

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Ctrl+C로 종료합니다...")
    finally:
        print("프로그램을 종료합니다...")
        robot_controller.terminate_gripper()
        robot_controller.stop_camera()
        cv2.destroyAllWindows()
        robot_controller.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        print("종료 완료.")


if __name__ == "__main__":
    main()