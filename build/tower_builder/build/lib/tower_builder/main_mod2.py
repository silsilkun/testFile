"""
🏗️ Smart Tower Builder - FINAL SYNC VERSION
===========================================
강사님 예제 코드 구조 기반 (싱글 스레드 블로킹 방식)
- 로봇이 움직이는 동안 프로그램이 기다립니다. (동기화 해결)
- 그리퍼는 '0'으로 꽉 잡습니다. (낙하 방지)
"""

import cv2
import rclpy
from rclpy.node import Node
import threading
import time
import numpy as np

# 사용자 정의 모듈
import DR_init
from tower_builder.gripper_drl_controller import GripperController
from tower_builder.camera import BlockDetectionSystem

# ============================================================
# ⚙️ [설정]
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 60, 60 # 속도 적절히 타협

# 좌표 보정 (지나님 설정 유지)
TRANSFORM_OFFSET_X = 685.0
TRANSFORM_OFFSET_Y = 20.0
CAMERA_Z_HEIGHT = 810.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        
        # 1. 카메라 시작
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            raise RuntimeError("Vision start failed")

        self.blocks = []
        self.target_stack_count = 0
        self.selected_queue = []
        self.stack_base_coords = None
        
        # 작업 중인지 확인하는 플래그
        self.is_processing = False 

        # 2. 그리퍼 초기화 (강사님 코드 스타일)
        self.gripper = None
        try:
            # 강사님 코드처럼 wait를 여기서 임포트해서 씀
            from DSR_ROBOT2 import wait
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            # 초기화 대기
            time.sleep(1.0) 
            if self.gripper.initialize():
                self.get_logger().info("✅ 그리퍼 객체 생성 완료")
                self.gripper.move(0) # 일단 한번 닫아봄
                time.sleep(1.0)
                self.gripper.move(700) # 벌림
        except Exception as e:
            self.get_logger().error(f"그리퍼 오류: {e}")

    def stop_camera(self):
        self.vision.stop()

    def terminate_gripper(self):
        if self.gripper: self.gripper.terminate()

    # ============================================================
    # 마우스 콜백 (여기서 로직 실행)
    # ============================================================
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.is_processing:
            print("⚠️ 작업 중입니다. 잠시만 기다려주세요!")
            return

        # 1. 블럭 선택 단계
        if len(self.selected_queue) < self.target_stack_count:
            block = self.vision.find_block_at(x, y, update=False)
            if block:
                if block in self.selected_queue: return
                self.selected_queue.append(block)
                block.selection_order = len(self.selected_queue)
                
                print(f"✅ 블럭 선택 [{len(self.selected_queue)}/{self.target_stack_count}]")
                
                if len(self.selected_queue) == self.target_stack_count:
                    print("\n🎯 블럭 선택 완료! [탑을 쌓을 바닥]을 클릭하세요.")

        # 2. 위치 지정 및 실행 단계
        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if point_3d is None or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패.")
                return

            cam_x_mm = point_3d[0] * 1000
            cam_y_mm = point_3d[1] * 1000

            # 좌표 변환 (지나님 설정)
            final_x = TRANSFORM_OFFSET_X + cam_y_mm
            final_y = cam_x_mm + TRANSFORM_OFFSET_Y
            
            self.stack_base_coords = (final_x, final_y)
            print(f"📍 타워 위치: ({final_x:.1f}, {final_y:.1f})")
            
            # [중요] 여기서 스레드 분리 안 함! 바로 실행! (강사님 방식)
            self.run_stacking_sequence()

    # ============================================================
    # 작업 시퀀스 (강사님 스타일 - movel + wait 조합)
    # ============================================================
    def run_stacking_sequence(self):
        self.is_processing = True
        print("\n🚀 로봇 작업 시퀀스 시작 (화면이 멈출 수 있습니다)")
        
        stack_x, stack_y = self.stack_base_coords
        BASE_Z = 152.0 # 바닥 높이
        BLOCK_H = 40.0 # 블럭 높이 간격

        # 필요한 함수들 여기서 임포트 (강사님 코드 참조)
        from DSR_ROBOT2 import movej, movel, wait, get_current_posx
        from DR_common2 import posj, posx

        try:
            # 1. 홈 정렬
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(2.0) # 강제 대기

            # 2. 적재 루프
            for i, block in enumerate(self.selected_queue):
                print(f"\n🏗️ [{i+1}층 작업 시작] ---------------------")

                # --- 좌표 계산 ---
                cam_x, cam_y, cam_z = block.center_3d_mm
                pick_x = TRANSFORM_OFFSET_X + cam_y
                pick_y = cam_x + TRANSFORM_OFFSET_Y
                pick_z = CAMERA_Z_HEIGHT - cam_z # 바닥면 기준 Z
                
                # 안전 높이 보정
                if pick_z < BASE_Z: pick_z = BASE_Z

                place_z = BASE_Z + (i * BLOCK_H)

                # --- 이동 로직 (강사님 코드 로직 + 우리 좌표) ---
                # 현재 회전값 유지
                cur_x_pos = get_current_posx()[0]
                rx, ry, rz = cur_x_pos[3], cur_x_pos[4], cur_x_pos[5]
                
                safe_z = 350.0

                # 1) Pick 상공 이동
                print("   🚀 [1] Pick 상공 이동")
                p_pick_ready = posx([pick_x, pick_y, safe_z, rx, ry, rz])
                movel(p_pick_ready, vel=VELOCITY, acc=ACC)
                wait(0.5)

                # 2) 그리퍼 벌리기
                if self.gripper: self.gripper.move(700)
                wait(0.5)

                # 3) 잡으러 내려가기
                print("   🔻 [2] Pick 하강")
                p_pick_down = posx([pick_x, pick_y, pick_z, rx, ry, rz])
                movel(p_pick_down, vel=VELOCITY/2, acc=ACC/2)
                wait(0.8) # 충분히 대기

                # 4) 잡기 (강사님처럼 0으로 꽉 잡기)
                print("   ✊ [3] 그립 (꽉!)")
                if self.gripper: self.gripper.move(0)
                wait(1.5) # 잡는 시간 확보

                # 5) 들어올리기
                print("   🔼 [4] Pick 상승")
                movel(p_pick_ready, vel=VELOCITY, acc=ACC)
                wait(0.5)

                # 6) Place 상공 이동
                print("   🚀 [5] Place 상공 이동")
                p_place_ready = posx([stack_x, stack_y, safe_z, rx, ry, rz])
                movel(p_place_ready, vel=VELOCITY, acc=ACC)
                wait(0.5)

                # 7) 놓으러 내려가기
                print("   🔻 [6] Place 하강")
                # 조금 더 위에서 놓도록 +15mm (안전)
                p_place_down = posx([stack_x, stack_y, place_z + 15.0, rx, ry, rz])
                movel(p_place_down, vel=VELOCITY/2, acc=ACC/2)
                wait(0.8)

                # 8) 놓기
                print("   🖐 [7] 놓기")
                if self.gripper: self.gripper.move(700)
                wait(1.0)

                # 9) 복귀
                print("   🔼 [8] 복귀")
                movel(p_place_ready, vel=VELOCITY, acc=ACC)
                wait(0.5)

            print("\n✨ 작업 완료! 홈 복귀.")
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(1.0)

        except Exception as e:
            self.get_logger().error(f"실행 중 오류: {e}")
        finally:
            self.selected_queue = []
            self.stack_base_coords = None
            self.is_processing = False
            print("🎉 다시 시작하려면 몇 층 쌓을지 입력하세요.")
            # target_stack_count를 0으로 만들어서 다시 입력받게 함
            self.target_stack_count = 0


    def process_and_render(self):
        # 작업 중이면 화면 갱신만 하고 리턴 (마우스 클릭 방지 등)
        if not self.vision.update(): return

        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()
        
        # UI 그리기
        for block in self.blocks:
            is_sel = (block in self.selected_queue)
            col = (0, 255, 255) if is_sel else (0, 255, 0)
            cv2.drawContours(display, [block.rotated_box], 0, col, 2)
            cx, cy = block.center_2d
            
            if hasattr(block, 'selection_order'):
                cv2.putText(display, f"#{block.selection_order}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if self.target_stack_count == 0: msg = "Input Number in Terminal"
        elif self.is_processing: msg = "Working... Please Wait..."
        elif len(self.selected_queue) < self.target_stack_count: msg = "Select Blocks..."
        elif self.stack_base_coords is None: msg = "Click Target Floor"
        else: msg = "Ready"
        
        cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Result", display)
        cv2.waitKey(1) # GUI 이벤트 처리용 (필수)


def main(args=None):
    rclpy.init(args=args)
    
    # 📌 강사님 코드와 동일하게 DSR 노드 생성
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    
    # 로봇 모드 설정 (안전빵)
    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except: pass

    robot = RobotControllerNode()

    # 📌 마우스 콜백 설정
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot.mouse_callback)

    print("\n" + "="*40)
    print("🏗️ Smart Tower Builder (FINAL SYNC VER)")
    print("="*40)

    try:
        # 📌 강사님 코드 스타일의 메인 루프 (Single Thread Loop)
        while rclpy.ok():
            # 1. 터미널 입력 (초기 상태일 때만)
            if robot.target_stack_count == 0 and not robot.is_processing:
                try:
                    # 입력을 기다릴 때도 spin_once를 돌려줘야 로봇 연결이 안 끊김
                    # 하지만 input()은 블로킹 함수라 여기서 멈추면 ROS가 끊길 수 있음.
                    # 그래서 타임아웃을 둔 select를 쓰거나 해야 하는데,
                    # 간단하게 그냥 non-blocking 체크는 어려우니
                    # 여기서는 그냥 입력 받을 때 멈추는 걸 감수함.
                    val = input("\n👉 몇 층 탑을 쌓으시겠습니까? (숫자) >> ")
                    cnt = int(val)
                    if cnt > 0:
                        robot.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 선택하세요.")
                except ValueError: pass
            
            # 2. 노드 스핀 (로봇 상태 업데이트 및 비전 처리)
            # 이걸 하나의 루프에서 다 돌립니다.
            rclpy.spin_once(robot, timeout_sec=0.01)
            rclpy.spin_once(dsr_node, timeout_sec=0.01)
            
            # 3. 화면 갱신
            robot.process_and_render()

    except KeyboardInterrupt:
        print("종료")
    finally:
        robot.terminate_gripper()
        robot.stop_camera()
        cv2.destroyAllWindows()
        robot.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()