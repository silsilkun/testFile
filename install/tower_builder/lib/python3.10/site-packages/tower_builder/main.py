"""
🏗️ Smart Tower Builder - Main (Queue System & Stacking)
======================================================
기능:
1. 시작 시 쌓을 층수(N) 입력
2. 화면에서 N개의 블럭을 클릭하여 순서 지정 (Queue)
3. 지정된 순서대로 블럭을 집어 고정된 위치에 적재
4. 블럭 크기별 그리퍼 힘 조절
"""

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
import time

# 사용자 정의 모듈 임포트
import DR_init
from tower_builder.gripper_drl_controller import GripperController
from tower_builder.camera import BlockDetectionSystem, Block

# ============================================================
# 로봇 설정
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 200, 100

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        self.get_logger().info("RobotControllerNode 시작")

        # =========================
        # 비전 시스템 초기화
        # =========================
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            self.get_logger().error("BlockDetectionSystem start() 실패")
            raise RuntimeError("Vision system start failed")

        self.blocks = []

        # =========================
        # 작업 큐(Queue) 시스템 변수
        # =========================
        self.target_stack_count = 0  # 목표 층수 (메인에서 입력받음)
        self.selected_queue = []     # 선택된 블럭 리스트
        self.is_working = False      # 로봇 작업 중 플래그

        # =========================
        # 로봇 및 그리퍼 초기화
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

            self.get_logger().info("그리퍼 초기화 완료 (Open)")
            self.gripper.move(0) # 초기 상태 오픈
        except Exception as e:
            self.get_logger().error(f"그리퍼 설정 중 오류: {e}")
            raise

    def stop_camera(self):
        if self.vision:
            self.vision.stop()

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    # ============================================================
    # [핵심] 마우스 클릭 콜백 (예약 시스템)
    # ============================================================
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 로봇이 작업 중이거나, 이미 목표 개수를 다 채웠으면 입력 무시
        if self.is_working:
            print("⚠️ 작업 중입니다. 잠시만 기다려주세요.")
            return

        if len(self.selected_queue) >= self.target_stack_count:
            print(f"⚠️ 이미 {self.target_stack_count}개를 모두 선택했습니다.")
            return

        # 클릭한 위치의 블럭 찾기
        block = self.vision.find_block_at(x, y, update=False)
        
        if block:
            # 중복 선택 방지
            if block in self.selected_queue:
                print("⚠️ 이미 선택한 블럭입니다.")
                return

            if not block.is_valid:
                print("⚠️ 깊이 정보가 정확하지 않은 블럭입니다. 다시 선택해주세요.")
                return

            # 큐에 추가
            self.selected_queue.append(block)
            current_idx = len(self.selected_queue)
            
            # 화면 표시용 순서 저장
            block.selection_order = current_idx
            
            print(f"✅ [선택 {current_idx}/{self.target_stack_count}] 블럭 등록 완료! (ID: {id(block)})")
            
            # 목표 개수 달성 시 작업 시작
            if len(self.selected_queue) == self.target_stack_count:
                print(f"\n🚀 {self.target_stack_count}개 선택 완료! 탑 쌓기를 시작합니다...")
                self.execute_stacking_sequence()
        else:
            print(f"❌ 빈 공간입니다. ({x}, {y})")
            # 디버깅용 깊이 찍기
            d = self.vision.get_depth_at(x, y)
            print(f"   (Depth: {d*100:.1f}cm)")

    # ============================================================
    # [핵심] 순차 실행 로직 (Sequencer)
    # ============================================================
    def execute_stacking_sequence(self):
        self.is_working = True  # 작업 시작
        
        # ----------------------------------------------------
        # 📍 타워 적재 위치 좌표 (사용 환경에 맞게 수정 필수!)
        # ----------------------------------------------------
        STACK_X = 400.0  
        STACK_Y = 0.0
        BASE_Z = 152.0   # 바닥 높이 (충돌 주의)
        BLOCK_H = 40.0   # 블럭 1개당 높이 증가량

        try:
            # 큐에 있는 블럭들을 순서대로 처리
            for i, block in enumerate(self.selected_queue):
                print(f"\n🏗️ [{i+1}번째 블럭] 작업 시작...")

                # 1. 픽업(Pick) 좌표 계산
                cam_x, cam_y, cam_z = block.center_3d_mm
                pick_x = 685 + cam_y
                pick_y = cam_x + 20
                pick_z = 810 - cam_z
                
                # 바닥 충돌 방지
                if pick_z <= 152: pick_z = 152

                # 블럭 폭 (그리퍼 제어용)
                width = min(block.real_width_mm, block.real_height_mm)

                # 2. 적재(Place) 높이 계산
                # 0번 인덱스 -> BASE_Z
                # 1번 인덱스 -> BASE_Z + 40mm ...
                place_z = BASE_Z + (i * BLOCK_H)

                print(f"   Pick: ({pick_x:.1f}, {pick_y:.1f}, {pick_z:.1f}) / Width: {width:.1f}")
                print(f"   Place: ({STACK_X}, {STACK_Y}, {place_z:.1f})")

                # 3. 로봇 이동 실행
                self.move_pick_and_place(pick_x, pick_y, pick_z, width, 
                                         STACK_X, STACK_Y, place_z)
            
            print("\n✨ 탑 쌓기 완료!")

            # ------------------------------------------------
            # [요청사항] 해체 작업 (주석 처리됨)
            # ------------------------------------------------
            # time.sleep(1.0)
            # self.dismantle_tower(self.target_stack_count, STACK_X, STACK_Y, BASE_Z, BLOCK_H)
            
        except Exception as e:
            self.get_logger().error(f"작업 중 치명적 오류: {e}")
        finally:
            # 작업 종료 후 초기화
            self.selected_queue = []  # 큐 비우기
            self.is_working = False
            print("🔄 초기화 완료. 다시 명령을 내릴 수 있습니다.")

    # ============================================================
    # [기능] 단일 블럭 Pick & Place
    # ============================================================
    def move_pick_and_place(self, px, py, pz, width, sx, sy, sz):
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj

        # 그리퍼 파워 설정
        grip_value = 550 # 기본 (Medium)
        if width < 40: grip_value = 750    # Small (꽉)
        elif width > 70: grip_value = 350  # Big (살살)

        current_pos = get_current_posx()[0]
        _, _, _, rx, ry, rz = current_pos
        
        safe_z = 300.0  # 이동 안전 높이

        # --- [PICK Sequence] ---
        # 1. 접근 (위)
        movel(posx([px, py, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)
        if self.gripper: self.gripper.move(0) # 오픈
        
        # 2. 하강
        movel(posx([px, py, pz, rx, ry, rz]), vel=VELOCITY/2, acc=ACC/2)
        wait(0.2)
        
        # 3. 잡기
        if self.gripper: self.gripper.move(grip_value)
        wait(1.2) 

        # 4. 상승
        movel(posx([px, py, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)

        # --- [PLACE Sequence] ---
        # 5. 적재 위치 이동 (위)
        movel(posx([sx, sy, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)

        # 6. 적재 위치 하강 (목표 높이 + 여유 10mm)
        movel(posx([sx, sy, sz + 15.0, rx, ry, rz]), vel=VELOCITY/2, acc=ACC/2)
        wait(0.5)

        # 7. 놓기
        if self.gripper: self.gripper.move(0)
        wait(0.5)

        # 8. 복귀
        movel(posx([sx, sy, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)
        
        # 9. 홈 포즈 (선택 사항 - 연속 동작을 위해 생략 가능하나 안정성을 위해 추가)
        # home_pose = posj(0, 0, 90, 0, 90, 0)
        # movej(home_pose, VELOCITY, ACC)

    # ============================================================
    # [기능] 탑 해체 (주석 처리됨)
    # ============================================================
    # def dismantle_tower(self, count, sx, sy, base_z, block_h):
    #     print("\n💣 탑 해체를 시작합니다...")
    #     from DSR_ROBOT2 import movel, wait
    #     from DR_common2 import posx
    # 
    #     # 위에서부터 아래로 해체 (역순 반복)
    #     for i in range(count - 1, -1, -1):
    #         current_z = base_z + (i * block_h)
    #         print(f"   🗑️ {i+1}층 해체 중... (높이: {current_z})")
    #         
    #         # 로직: Pick(탑 위치) -> Place(쓰레기통 or 랜덤 위치)
    #         # (이곳에 move_pick_and_place와 유사한 로직 구현 필요)
    #         pass

    # ============================================================
    # 화면 렌더링
    # ============================================================
    def process_and_render(self):
        # 트랙바 값 반영
        self.vision.config.threshold = cv2.getTrackbarPos("Threshold", "Control")
        self.vision.config.min_area = cv2.getTrackbarPos("Min Area", "Control")
        self.vision.config.max_area = cv2.getTrackbarPos("Max Area", "Control")

        # 비전 업데이트
        if not self.vision.update():
            return

        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()

        # ROI 박스 그리기
        cfg = self.vision.config
        cv2.rectangle(display, (cfg.roi_x, cfg.roi_y), 
                     (cfg.roi_x + cfg.roi_w, cfg.roi_y + cfg.roi_h), (0, 0, 255), 2)

        # 블록 그리기
        for block in self.blocks:
            # 큐에 포함된 블럭인지 확인
            is_selected = (block in self.selected_queue)
            
            color_draw = (0, 255, 255) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2

            cv2.drawContours(display, [block.rotated_box], 0, color_draw, thickness)

            cx, cy = block.center_2d
            
            # 정보 텍스트
            if block.is_valid:
                info_txt = f"{block.real_width_mm:.0f}x{block.real_height_mm:.0f}"
                cv2.putText(display, info_txt, (cx - 30, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # [중요] 선택된 순서 번호 표시 (①, ② ...)
            if hasattr(block, 'selection_order'):
                order_txt = f"#{block.selection_order}"
                cv2.putText(display, order_txt, (cx - 15, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 상태 표시
        status_txt = f"Selected: {len(self.selected_queue)} / {self.target_stack_count}"
        cv2.putText(display, status_txt, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if self.is_working:
             cv2.putText(display, "WORKING...", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Result", display)
        
        # Depth 화면
        if self.vision.last_depth is not None:
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(self.vision.last_depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_display)


def main(args=None):
    rclpy.init(args=args)

    # DSR 노드 생성
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except ImportError as e:
        print(f"DSR 라이브러리 로드 실패: {e}")
        return

    robot_controller = RobotControllerNode()

    # ==========================================
    # [입력] 사용자에게 쌓을 개수 입력 받기
    # ==========================================
    print("\n" + "="*40)
    print("🏗️ Smart Tower Builder v2.0")
    print("="*40)
    try:
        val = input("👉 몇 층 탑을 쌓으시겠습니까? (숫자 입력) >> ")
        robot_controller.target_stack_count = int(val)
        print(f"\n✅ {robot_controller.target_stack_count}개의 블럭을 화면에서 순서대로 클릭하세요!")
    except ValueError:
        print("⚠️ 숫자가 아닙니다. 기본값 3개로 설정합니다.")
        robot_controller.target_stack_count = 3
    print("="*40 + "\n")

    # 창 설정
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot_controller.mouse_callback)

    cv2.namedWindow("Control")
    cfg = robot_controller.vision.config
    cv2.createTrackbar("Threshold", "Control", cfg.threshold, 255, lambda x: None)
    cv2.createTrackbar("Min Area", "Control", cfg.min_area, 5000, lambda x: None)
    cv2.createTrackbar("Max Area", "Control", cfg.max_area, 30000, lambda x: None)

    try:
        while rclpy.ok():
            rclpy.spin_once(robot_controller, timeout_sec=0.001)
            rclpy.spin_once(dsr_node, timeout_sec=0.001)

            robot_controller.process_and_render()

            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

    except KeyboardInterrupt:
        print("종료 요청됨...")
    finally:
        print("시스템 종료 중...")
        robot_controller.terminate_gripper()
        robot_controller.stop_camera()
        cv2.destroyAllWindows()
        robot_controller.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()