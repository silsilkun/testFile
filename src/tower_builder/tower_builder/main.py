"""
🏗️ Smart Tower Builder - TURTLE MODE (Slow & Accurate)
=====================================================
[안정화 패치]
1. 속도/가속도 대폭 하향 (50/30) -> 씹힘 방지
2. 동작 사이 wait 시간 2배 이상 증가 -> 확실한 동작 보장
3. 터미널 명령어 전원 공급 유지
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import threading
import time
import os

# 사용자 정의 모듈
import DR_init
from tower_builder.gripper_drl_controller import GripperController
from tower_builder.camera import BlockDetectionSystem, Block

# ============================================================
# ⚙️ [설정] 속도를 아주 느리게 설정 (안정성 최우선)
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"

# 🐢 거북이 세팅: 답답해도 참으세요! 무조건 성공합니다.
VELOCITY = 50   # 기존 150 -> 50
ACC = 30        # 기존 80 -> 30

# 📍 좌표 보정
TRANSFORM_OFFSET_X = 685.0
TRANSFORM_OFFSET_Y = 20.0
CAMERA_Z_HEIGHT = 810.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            raise RuntimeError("Vision start failed")

        self.blocks = []
        self.target_stack_count = 0
        self.selected_queue = []
        self.stack_base_coords = None
        self.is_working = False

        self.gripper = None
        try:
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            time.sleep(1.0)
            if self.gripper.initialize():
                self.get_logger().info("✅ 그리퍼 컨트롤러 연결됨")
        except Exception as e:
            self.get_logger().error(f"그리퍼 오류: {e}")

    def stop_camera(self):
        self.vision.stop()

    def terminate_gripper(self):
        if self.gripper: self.gripper.terminate()

    # 터미널 명령어로 전원 켜기
    def turn_on_power_via_terminal(self):
        print("\n⚡ [시스템] 그리퍼 전원(24V) 공급 명령 전송...")
        cmd = f"ros2 service call /{ROBOT_ID}/tool/set_tool_voltage dsr_msgs/srv/SetToolVoltage \"{{voltage: 24}}\""
        os.system(cmd)
        time.sleep(2.0) # 전원 켜지고 충분히 대기

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.is_working: return

        if len(self.selected_queue) < self.target_stack_count:
            block = self.vision.find_block_at(x, y, update=False)
            if block:
                if block in self.selected_queue: return
                self.selected_queue.append(block)
                block.selection_order = len(self.selected_queue)
                w = min(block.real_width_mm, block.real_height_mm)
                print(f"✅ 블럭 선택 [{len(self.selected_queue)}/{self.target_stack_count}] (크기: {w:.1f}mm)")
                if len(self.selected_queue) == self.target_stack_count:
                    print("\n🎯 블럭 선택 완료! [탑을 쌓을 바닥]을 클릭하세요.")

        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if point_3d is None or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패.")
                return
            cam_x_mm = point_3d[0] * 1000
            cam_y_mm = point_3d[1] * 1000
            final_x = TRANSFORM_OFFSET_X + cam_y_mm
            final_y = cam_x_mm + TRANSFORM_OFFSET_Y
            self.stack_base_coords = (final_x, final_y)
            print(f"📍 타워 위치: ({final_x:.1f}, {final_y:.1f})")
            
            worker = threading.Thread(target=self.execute_stacking_sequence)
            worker.daemon = True
            worker.start()

    def execute_stacking_sequence(self):
        self.is_working = True
        print("\n🚀 로봇 작업 시퀀스 시작 (안정 모드)")
        
        stack_x, stack_y = self.stack_base_coords
        BASE_Z = 152.0
        BLOCK_H = 40.0

        from DSR_ROBOT2 import movej, movel, wait
        from DR_common2 import posj, posx

        try:
            # 1. 전원 켜기
            self.turn_on_power_via_terminal()

            # 2. 그리퍼 워밍업 (천천히)
            if self.gripper:
                print("✊ 그리퍼 동작 테스트...")
                self.gripper.move(0); wait(1.0)
                self.gripper.move(800); wait(1.0)
                self.gripper.move(0); wait(1.0)

            # 3. 홈 정렬
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(2.0) # 충분히 멈출 때까지 대기

            for i, block in enumerate(self.selected_queue):
                print(f"\n🏗️ [{i+1}층 작업 시작] ===================")

                cam_x, cam_y, cam_z = block.center_3d_mm
                pick_x = TRANSFORM_OFFSET_X + cam_y
                pick_y = cam_x + TRANSFORM_OFFSET_Y
                pick_z = CAMERA_Z_HEIGHT - cam_z
                if pick_z < BASE_Z: pick_z = BASE_Z
                
                print(f"   📍 목표 Pick 좌표: X={pick_x:.1f}, Y={pick_y:.1f}, Z={pick_z:.1f}")

                width = min(block.real_width_mm, block.real_height_mm)
                place_z = BASE_Z + (i * BLOCK_H)

                target_open = 0; target_close = 500
                if width <= 35: target_open = 300; target_close = 850
                elif width <= 45: target_open = 200; target_close = 600
                else: target_open = 0; target_close = 350

                # ------------------------------------------------
                # [안정적 이동 로직] Wait 대폭 추가
                # ------------------------------------------------
                
                # 1. [접근] movej
                print("   🚀 [1] Pick 상공으로 이동...")
                approach_pos = posx([pick_x, pick_y, 350.0, 90.0, 180.0, 0.0])
                movej(approach_pos, vel=VELOCITY, acc=ACC)
                wait(1.5) # 도착 후 확실히 멈춤

                # 2. 그리퍼 준비
                if self.gripper: self.gripper.move(target_open)
                wait(0.5)

                # 3. [하강] movel
                print("   🔻 [2] 하강...")
                pick_pos = posx([pick_x, pick_y, pick_z, 90.0, 180.0, 0.0])
                movel(pick_pos, vel=VELOCITY/2, acc=ACC/2) # 하강은 더 천천히
                wait(1.5) # 확실히 정지

                # 4. [잡기]
                print(f"   ✊ [3] 그립 (강도: {target_close})")
                if self.gripper: self.gripper.move(target_close)
                wait(2.0) # 잡는 시간 2초 부여 (확실하게)

                # 5. [상승]
                print("   🔼 [4] 상승...")
                movel(approach_pos, vel=VELOCITY, acc=ACC)
                wait(1.0)

                # 6. [Place 이동] movej
                print("   🚀 [5] 적재 위치로 이동...")
                stack_approach = posx([stack_x, stack_y, 350.0, 90.0, 180.0, 0.0])
                movej(stack_approach, vel=VELOCITY, acc=ACC)
                wait(1.5)
                
                # 7. [적재 하강]
                print("   🔻 [6] 적재 하강...")
                stack_pos = posx([stack_x, stack_y, place_z + 15.0, 90.0, 180.0, 0.0])
                movel(stack_pos, vel=VELOCITY/2, acc=ACC/2)
                wait(1.5)

                # 8. [놓기]
                print("   🖐 [7] 놓기")
                if self.gripper: self.gripper.move(0)
                wait(1.5) # 놓는 시간 충분히

                # 9. [복귀]
                movel(stack_approach, vel=VELOCITY, acc=ACC)
                wait(1.0)

            print("\n✨ 작업 완료! 홈 복귀.")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)

        except Exception as e:
            self.get_logger().error(f"실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.selected_queue = []
            self.stack_base_coords = None
            self.is_working = False

    def process_and_render(self):
        self.vision.config.threshold = cv2.getTrackbarPos("Threshold", "Control")
        self.vision.config.min_area = cv2.getTrackbarPos("Min Area", "Control")
        self.vision.config.max_area = cv2.getTrackbarPos("Max Area", "Control")

        if not self.vision.update(): return

        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()
        c = self.vision.config
        cv2.rectangle(display, (c.roi_x, c.roi_y), (c.roi_x+c.roi_w, c.roi_y+c.roi_h), (0,0,255), 2)

        for block in self.blocks:
            is_sel = (block in self.selected_queue)
            col = (0, 255, 255) if is_sel else (0, 255, 0)
            cv2.drawContours(display, [block.rotated_box], 0, col, 2)
            cx, cy = block.center_2d
            w_mm = min(block.real_width_mm, block.real_height_mm)
            cv2.putText(display, f"{w_mm:.0f}mm", (cx-20, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if hasattr(block, 'selection_order'):
                cv2.putText(display, f"#{block.selection_order}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if self.target_stack_count == 0: msg = "Input number in terminal"
        elif len(self.selected_queue) < self.target_stack_count: msg = "Select Blocks..."
        elif self.stack_base_coords is None: msg = ">> Click Target Floor <<"
        else: msg = "Auto Stacking..."
        cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Result", display)
        
        if self.vision.last_depth is not None:
            d_view = cv2.applyColorMap(cv2.convertScaleAbs(self.vision.last_depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Depth", d_view)


def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except: pass

    robot = RobotControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(dsr_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot.mouse_callback)
    cv2.namedWindow("Control")
    cfg = robot.vision.config
    cv2.createTrackbar("Threshold", "Control", cfg.threshold, 255, lambda x: None)
    cv2.createTrackbar("Min Area", "Control", cfg.min_area, 5000, lambda x: None)
    cv2.createTrackbar("Max Area", "Control", cfg.max_area, 30000, lambda x: None)

    print("\n" + "="*40)
    print("🏗️ Smart Tower Builder (TURTLE MODE)")
    print("="*40)

    try:
        while rclpy.ok():
            if robot.target_stack_count == 0:
                try:
                    val = input("\n👉 몇 층 탑을 쌓으시겠습니까? (숫자) >> ")
                    cnt = int(val)
                    if cnt > 0:
                        robot.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 선택하세요.")
                except ValueError: pass
                continue
            robot.process_and_render()
            if not robot.is_working and robot.target_stack_count > 0 and robot.stack_base_coords is not None:
                if len(robot.selected_queue) == 0:
                    robot.target_stack_count = 0
                    robot.stack_base_coords = None
                    print("\n🎉 완료! 다시 시작합니다.")
            if cv2.waitKey(1) & 0xFF == 27: break
    except KeyboardInterrupt: print("종료")
    finally:
        robot.terminate_gripper()
        robot.stop_camera()
        cv2.destroyAllWindows()
        executor.shutdown()
        robot.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()