"""
🏗️ Smart Tower Builder - FINAL SAFETY UPDATE
===========================================
[수정 내용]
1. 강제 종료(Ctrl+C) 시 '무조건 홈 정렬' 기능 추가
   - 프로그램이 꺼지기 직전, 로봇을 안전한 대기 자세로 이동시킴
2. 기존 로직(높이 보정, 좌표 변환, 스레드) 모두 유지
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
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
VELOCITY, ACC = 60, 60

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

        # 그리퍼 초기화
        self.gripper = None
        try:
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            time.sleep(1.0)
            if self.gripper.initialize():
                self.get_logger().info("✅ 그리퍼 연결 성공")
                self.gripper.move(0) 
                time.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f"그리퍼 오류: {e}")

    def stop_camera(self):
        self.vision.stop()

    def terminate_gripper(self):
        if self.gripper: self.gripper.terminate()

    # ============================================================
    # [좌표 변환]
    # ============================================================
    def convert_camera_to_robot(self, cam_x_mm, cam_y_mm, cam_z_mm):
        final_x = 690 + cam_y_mm 
        final_y = cam_x_mm
        final_z = 823 - cam_z_mm
        if final_z <= 150.0: final_z = 150.0
        return final_x, final_y, final_z

    # ============================================================
    # 마우스 콜백
    # ============================================================
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.is_working:
            print("⚠️ 로봇이 작업 중입니다! 잠시만 기다려주세요.")
            return

        # 1. 블럭 선택
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

        # 2. 위치 지정 및 실행
        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if point_3d is None or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패. 다시 클릭해주세요.")
                return

            rs_x_mm = point_3d[0] * 1000
            rs_y_mm = point_3d[1] * 1000
            rs_z_mm = point_3d[2] * 1000

            final_x, final_y, final_z = self.convert_camera_to_robot(rs_x_mm, rs_y_mm, rs_z_mm)
            
            self.stack_base_coords = (final_x, final_y, final_z)
            print(f"📍 타워 위치(Base): X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}")
            
            # 스레드 실행
            worker = threading.Thread(target=self.execute_stacking_sequence)
            worker.daemon = True
            worker.start()

    # ============================================================
    # 실행 시퀀스
    # ============================================================
    def execute_stacking_sequence(self):
        self.is_working = True
        print("\n🚀 로봇 작업 시퀀스 시작")
        
        stack_x, stack_y, stack_base_z = self.stack_base_coords
        current_stack_height = 0.0 

        from DSR_ROBOT2 import movel, movej, get_current_posx, wait
        from DR_common2 import posx, posj

        try:
            # 1. 홈 정렬
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

            # 2. 적재 루프
            for i, block in enumerate(self.selected_queue):
                # 블럭 크기
                w = min(block.real_width_mm, block.real_height_mm)

                print(f"\n🏗️ [{i+1}층 작업 시작] (블럭 높이: {w:.1f}mm) =======")

                cam_x, cam_y, cam_z = block.center_3d_mm
                pick_x, pick_y, pick_z = self.convert_camera_to_robot(cam_x, cam_y, cam_z)
                
                # Place 높이 계산 (이전 블럭 높이 + 내 블럭 높이 + 여유 2mm)
                place_z = stack_base_z + current_stack_height + w + 2.0

                # 접근 각도 계산 및 반전
                cur_pos = get_current_posx()[0]
                cur_x, cur_y = cur_pos[0], cur_pos[1]
                cur_rx, cur_ry = cur_pos[3], cur_pos[4]

                dx, dy = pick_x - cur_x, pick_y - cur_y
                if abs(dx) > abs(dy): approach_axis = "x+" if dx > 0 else "x-"
                else: approach_axis = "y+" if dy > 0 else "y-"

                if approach_axis == "x+":   Rz_target = 0.0
                elif approach_axis == "x-": Rz_target = 180.0
                elif approach_axis == "y+": Rz_target = 90.0
                elif approach_axis == "y-": Rz_target = -90.0
                
                print(f"   🧭 접근 방향: {approach_axis} -> Rz: {Rz_target}")

                SAFE_Z = 350.0 
                
                val_open = 0 
                if w >= 45: val_close = 650
                elif w >= 35: val_close = 550
                else: val_close = 400

                # ================= [PICK] =================
                print("   🚀 [1] 상공 이동")
                p_high = posx([pick_x, pick_y, SAFE_Z, cur_rx, cur_ry, Rz_target])
                movel(p_high, vel=VELOCITY, acc=ACC)
                wait(3.5)

                if self.gripper: self.gripper.move(val_open)
                wait(2)

                print("   🔻 [2] 목표 상공 진입")
                p_ready = posx([pick_x, pick_y, pick_z + 50, cur_rx, cur_ry, Rz_target])
                movel(p_ready, vel=VELOCITY, acc=ACC)
                wait(2)

                print(f"   🔻 [3] Pick 하강 (Z={pick_z:.1f})")
                p_pick = posx([pick_x, pick_y, pick_z, cur_rx, cur_ry, Rz_target])
                movel(p_pick, vel=VELOCITY/2, acc=ACC/2)
                wait(3)

                print(f"   ✊ [4] 그립 ({val_close})")
                if self.gripper: self.gripper.move(val_close)
                wait(3)

                print("   🔼 [5] 상승")
                movel(p_high, vel=VELOCITY, acc=ACC)
                wait(3)

                # ================= [PLACE] =================
                print(f"   🚀 [6] Place 상공 이동 (목표 Z: {place_z:.1f})")
                p_place_high = posx([stack_x, stack_y, SAFE_Z, cur_rx, cur_ry, Rz_target])
                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(3)

                print(f"   🔻 [7] Place 하강")
                p_place = posx([stack_x, stack_y, place_z, cur_rx, cur_ry, Rz_target])
                movel(p_place, vel=VELOCITY/2, acc=ACC/2)
                wait(3)

                print("   🖐 [8] 놓기")
                if self.gripper: self.gripper.move(val_open)
                wait(4)

                print("   🔼 [9] 복귀")
                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(5)
                
                # 높이 누적
                current_stack_height += w
                print(f"   📈 스택 높이 업데이트: {current_stack_height:.1f}mm")

            print("\n✨ 모든 작업 완료! 홈으로 이동.")
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

        except Exception as e:
            self.get_logger().error(f"작업 중 오류 발생: {e}")
        finally:
            self.selected_queue = []
            self.stack_base_coords = None
            self.is_working = False
            self.target_stack_count = 0 
            print("🎉 완료! 다시 시작하려면 터미널을 확인하세요.")

    def process_and_render(self):
        if not self.vision.update(): return

        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()
        
        for block in self.blocks:
            is_sel = (block in self.selected_queue)
            col = (0, 255, 255) if is_sel else (0, 255, 0)
            cv2.drawContours(display, [block.rotated_box], 0, col, 2)
            
            if hasattr(block, 'selection_order'):
                cx, cy = block.center_2d
                cv2.putText(display, f"#{block.selection_order}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if self.target_stack_count == 0: msg = "Input Number in Terminal"
        elif self.is_working: msg = "Robot Working..."
        elif len(self.selected_queue) < self.target_stack_count: msg = f"Select Blocks ({len(self.selected_queue)}/{self.target_stack_count})"
        elif self.stack_base_coords is None: msg = "Click Target Floor"
        else: msg = "Ready"
        
        cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Result", display)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    
    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except: pass

    robot = RobotControllerNode()

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot.mouse_callback)

    print("\n" + "="*40)
    print("🏗️ Smart Tower Builder (SAFETY UPDATE)")
    print("="*40)

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(dsr_node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            if robot.target_stack_count == 0 and not robot.is_working:
                try:
                    val = input("\n👉 몇 층 탑을 쌓으시겠습니까? (숫자) >> ")
                    cnt = int(val)
                    if cnt > 0:
                        robot.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 화면에서 선택하세요.")
                except ValueError: pass
            
            robot.process_and_render()
            if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        print("\n🛑 사용자 종료 요청 (Ctrl+C)")
    
    # =======================================================
    # [강제 종료 시 무조건 홈 위치 정렬 로직]
    # =======================================================
    finally:
        print("\n🚨 프로그램 종료 절차 시작! 홈 위치로 복귀합니다...")
        try:
            from DSR_ROBOT2 import movej, wait
            from DR_common2 import posj
            
            # 홈 위치 정의
            home_pose = posj(0, 0, 90, 0, 90, 0)
            
            # 안전하게 이동 (속도 살짝 줄임)
            movej(home_pose, vel=50, acc=50)
            wait(3.0)
            print("✅ 홈 위치 복귀 완료. 안전하게 종료합니다.")
            
        except Exception as e:
            print(f"⚠️ 홈 복귀 실패 (이미 연결이 끊겼거나 오류 발생): {e}")

        robot.terminate_gripper()
        robot.stop_camera()
        cv2.destroyAllWindows()
        executor.shutdown()
        robot.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()