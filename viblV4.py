"""
🏗️ Block Detection System - RealSense (v6.2)
============================================
결정(Decided) 블록 시스템

변경사항:
    - CSV/JSON 저장 기능 삭제
    - 's' 키: 클릭한 블록들을 "결정" 상태로 확정 (덮어쓰기)
    - GUI 상태 표시: Clicked / Decided 개수

사용 예시:
    with BlockDetectionSystem() as system:
        system.run_debug()
        
        # 사용자가 블록 클릭 → 's' 키로 결정
        # 결정된 블록들 확인
        decided = system.get_decided_blocks()
        print(f"최종 선택: {len(decided)}개")
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime


# ============================================================
# 데이터 클래스
# ============================================================

@dataclass
class Block:
    """감지된 블록 정보를 담는 데이터 클래스"""
    
    # 기본 정보
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center_2d: Tuple[int, int]       # 픽셀 좌표 (x, y)
    contour: np.ndarray = field(compare=False, repr=False)
    rotated_box: np.ndarray = field(compare=False, repr=False)
    
    # 형태 정보
    area: float = 0.0
    aspect_ratio: float = 0.0
    solidity: float = 0.0
    angle: float = 0.0  # 회전 각도 (도)
    
    # 3D 정보
    center_3d: Optional[Tuple[float, float, float]] = None  # (X, Y, Z) 미터
    depth: float = 0.0  # 미터
    
    # 실제 크기 (mm)
    real_width_mm: float = 0.0
    real_height_mm: float = 0.0
    
    # 메타 정보
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    click_order: int = 0  # 클릭 순서
    
    # -------------------- 편의 속성 --------------------
    
    @property
    def side_length_mm(self) -> float:
        """한 변의 평균 길이 (mm)"""
        return (self.real_width_mm + self.real_height_mm) / 2
    
    @property
    def depth_cm(self) -> float:
        """깊이 (cm)"""
        return self.depth * 100
    
    @property
    def depth_mm(self) -> float:
        """깊이 (mm)"""
        return self.depth * 1000
    
    @property
    def center_3d_mm(self) -> Optional[Tuple[float, float, float]]:
        """3D 좌표 (mm 단위)"""
        if self.center_3d is None:
            return None
        x, y, z = self.center_3d
        return (x * 1000, y * 1000, z * 1000)
    
    @property
    def is_valid(self) -> bool:
        """유효한 깊이 정보가 있는지"""
        return self.depth > 0
    
    def __str__(self) -> str:
        if self.is_valid:
            return (f"Block(order={self.click_order}, center={self.center_2d}, "
                    f"depth={self.depth_cm:.1f}cm, "
                    f"size={self.side_length_mm:.1f}mm)")
        return f"Block(order={self.click_order}, center={self.center_2d}, no depth)"


# ============================================================
# 카메라 클래스
# ============================================================

class RealSenseCamera:
    """RealSense 카메라 제어 클래스"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._intrinsics: Optional[rs.intrinsics] = None
        self._depth_scale: float = 0.001
        self._is_running: bool = False
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def intrinsics(self) -> Optional[rs.intrinsics]:
        return self._intrinsics
    
    @property
    def depth_scale(self) -> float:
        return self._depth_scale
    
    def start(self) -> bool:
        """카메라 시작"""
        if self._is_running:
            return True
            
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, 
                               rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, 
                               rs.format.z16, self.fps)
            
            profile = self._pipeline.start(config)
            self._align = rs.align(rs.stream.color)
            
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            
            depth_stream = profile.get_stream(rs.stream.depth)
            self._intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print("📷 카메라 초기화 중...")
            for _ in range(30):
                self._pipeline.wait_for_frames()
            
            self._is_running = True
            print(f"✅ RealSense 시작! ({self.width}x{self.height})")
            return True
            
        except Exception as e:
            print(f"❌ 카메라 오류: {e}")
            return False
    
    def stop(self):
        """카메라 정지"""
        if self._pipeline and self._is_running:
            self._pipeline.stop()
            self._is_running = False
            print("📷 카메라 정지")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """정렬된 컬러/뎁스 프레임 반환"""
        if not self._is_running:
            return None, None
            
        try:
            frames = self._pipeline.wait_for_frames()
            aligned = self._align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception:
            return None, None
    
    def get_depth_at(self, x: int, y: int, depth_image: np.ndarray, 
                     use_sampling: bool = True) -> float:
        """특정 픽셀의 깊이값 반환 (미터)"""
        x, y = int(x), int(y)
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        raw_depth = depth_image[y, x]
        depth_m = raw_depth * self._depth_scale
        
        if depth_m > 0.05:
            return depth_m
        
        if not use_sampling:
            return 0.0
        
        offsets = [
            (-5, 0), (5, 0), (0, -5), (0, 5),
            (-10, 0), (10, 0), (0, -10), (0, 10),
            (-5, -5), (5, -5), (-5, 5), (5, 5),
            (-15, 0), (15, 0), (0, -15), (0, 15),
        ]
        
        valid_depths = []
        for dx, dy in offsets:
            sx, sy = x + dx, y + dy
            if 0 <= sx < self.width and 0 <= sy < self.height:
                raw = depth_image[sy, sx]
                d = raw * self._depth_scale
                if 0.05 < d < 3.0:
                    valid_depths.append(d)
        
        if valid_depths:
            valid_depths.sort()
            return valid_depths[len(valid_depths) // 2]
        
        return 0.0
    
    def pixel_to_3d(self, x: int, y: int, depth_image: np.ndarray
                   ) -> Optional[Tuple[float, float, float]]:
        """픽셀 좌표를 3D 좌표로 변환 (미터)"""
        depth = self.get_depth_at(x, y, depth_image)
        
        if depth <= 0 or self._intrinsics is None:
            return None
        
        point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [x, y], depth)
        return (point[0], point[1], depth)
    
    def calc_real_size(self, width_px: float, height_px: float, 
                       depth: float) -> Tuple[float, float]:
        """픽셀 크기를 실제 크기(mm)로 변환"""
        if depth <= 0 or self._intrinsics is None:
            return (0.0, 0.0)
        
        real_w = (width_px * depth * 1000) / self._intrinsics.fx
        real_h = (height_px * depth * 1000) / self._intrinsics.fy
        return (real_w, real_h)


# ============================================================
# 감지기 클래스
# ============================================================

@dataclass
class DetectorConfig:
    """감지기 설정"""
    threshold: int = 200
    
    roi_x: int = 190
    roi_y: int = 140
    roi_w: int = 230
    roi_h: int = 180
    
    min_area: int = 90
    max_area: int = 4000
    
    min_aspect: float = 0.5
    max_aspect: float = 3.0
    min_solidity: float = 0.7
    
    min_depth: float = 0.1
    max_depth: float = 2.0


class BlockDetector:
    """블록 감지기"""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._binary_view: Optional[np.ndarray] = None
    
    @property
    def binary_view(self) -> Optional[np.ndarray]:
        return self._binary_view
    
    @property
    def roi(self) -> Tuple[int, int, int, int]:
        c = self.config
        return (c.roi_x, c.roi_y, c.roi_w, c.roi_h)
    
    def detect(self, frame: np.ndarray, depth_image: np.ndarray,
               camera: RealSenseCamera) -> List[Block]:
        """프레임에서 블록 감지"""
        cfg = self.config
        blocks = []
        
        roi = frame[cfg.roi_y:cfg.roi_y+cfg.roi_h, 
                   cfg.roi_x:cfg.roi_x+cfg.roi_w]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, cfg.threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self._binary_view = binary
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            block = self._process_contour(cnt, depth_image, camera)
            if block is not None:
                blocks.append(block)
        
        return blocks
    
    def _process_contour(self, cnt: np.ndarray, depth_image: np.ndarray,
                         camera: RealSenseCamera) -> Optional[Block]:
        """단일 컨투어 처리"""
        cfg = self.config
        
        area = cv2.contourArea(cnt)
        if not (cfg.min_area < area < cfg.max_area):
            return None
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (_, _), (w, h), angle = rect
        
        if h == 0 or w == 0:
            return None
        
        aspect = max(w, h) / min(w, h)
        if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
            return None
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None
        solidity = area / hull_area
        if solidity < cfg.min_solidity:
            return None
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if not (3 <= len(approx) <= 8):
            return None
        
        box_global = box.copy()
        box_global[:, 0] += cfg.roi_x
        box_global[:, 1] += cfg.roi_y
        
        cnt_global = cnt.copy()
        cnt_global[:, :, 0] += cfg.roi_x
        cnt_global[:, :, 1] += cfg.roi_y
        
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + cfg.roi_x
        cy = int(M["m01"] / M["m00"]) + cfg.roi_y
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        block = Block(
            bbox=(x + cfg.roi_x, y + cfg.roi_y, bw, bh),
            center_2d=(cx, cy),
            contour=cnt_global,
            rotated_box=box_global,
            area=area,
            aspect_ratio=aspect,
            solidity=solidity,
            angle=angle
        )
        
        point_3d = camera.pixel_to_3d(cx, cy, depth_image)
        
        if point_3d:
            block.center_3d = point_3d
            block.depth = point_3d[2]
            
            if cfg.min_depth < block.depth < cfg.max_depth:
                real_w, real_h = camera.calc_real_size(w, h, block.depth)
                block.real_width_mm = real_w
                block.real_height_mm = real_h
        
        return block


# ============================================================
# 통합 시스템 클래스
# ============================================================

class BlockDetectionSystem:
    """
    블록 감지 통합 시스템 v6.2
    
    사용 예시:
        with BlockDetectionSystem() as system:
            system.run_debug()
            
            # 사용자가 블록 클릭 → 's' 키로 결정
            decided = system.get_decided_blocks()
            print(f"최종 선택: {len(decided)}개")
    """
    
    def __init__(self, 
                 camera_width: int = 640,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 config: Optional[DetectorConfig] = None):
        self._camera = RealSenseCamera(camera_width, camera_height, camera_fps)
        self._detector = BlockDetector(config)
        
        self._last_frame: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_blocks: List[Block] = []
        
        # 클릭한 블록 (임시)
        self._clicked_blocks: List[Block] = []
        
        # 결정된 블록 (확정)
        self._decided_blocks: List[Block] = []
    
    def __enter__(self) -> "BlockDetectionSystem":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def start(self) -> bool:
        return self._camera.start()
    
    def stop(self):
        self._camera.stop()
        cv2.destroyAllWindows()
    
    @property
    def is_running(self) -> bool:
        return self._camera.is_running
    
    @property
    def config(self) -> DetectorConfig:
        return self._detector.config
    
    @config.setter
    def config(self, value: DetectorConfig):
        self._detector.config = value
    
    @property
    def camera(self) -> RealSenseCamera:
        return self._camera
    
    @property
    def detector(self) -> BlockDetector:
        return self._detector
    
    def update(self) -> bool:
        """새 프레임을 가져와서 블록 감지 수행"""
        color, depth = self._camera.get_frames()
        if color is None:
            return False
        
        self._last_frame = color
        self._last_depth = depth
        self._last_blocks = self._detector.detect(color, depth, self._camera)
        return True
    
    def get_blocks(self, update: bool = True) -> List[Block]:
        """감지된 블록 리스트 반환"""
        if update:
            self.update()
        return self._last_blocks.copy()
    
    def get_valid_blocks(self, update: bool = True) -> List[Block]:
        """유효한 깊이 정보가 있는 블록만 반환"""
        blocks = self.get_blocks(update)
        return [b for b in blocks if b.is_valid]
    
    # -------------------- 블록 관리 --------------------
    
    def get_clicked_blocks(self) -> List[Block]:
        """클릭한 블록 리스트 반환 (임시)"""
        return self._clicked_blocks.copy()
    
    def get_decided_blocks(self) -> List[Block]:
        """결정된 블록 리스트 반환 (확정)"""
        return self._decided_blocks.copy()
    
    def decide_blocks(self):
        """
        클릭한 블록들을 결정 상태로 확정
        (기존 decided_blocks는 덮어쓰기)
        """
        if not self._clicked_blocks:
            print("⚠️  클릭한 블록이 없습니다")
            return
        
        self._decided_blocks = self._clicked_blocks.copy()
        print(f"\n✅ {len(self._decided_blocks)}개 블록 결정됨!")
        self.print_decided_blocks()
        self._clicked_blocks.clear()
    
    def clear_clicked_blocks(self):
        """클릭한 블록 리스트만 초기화"""
        self._clicked_blocks.clear()
        print("🗑️  클릭 블록 리스트 초기화됨")
    
    def reset_all(self):
        """모든 블록 리스트 초기화 (clicked + decided)"""
        self._clicked_blocks.clear()
        self._decided_blocks.clear()
        print("🗑️  모든 블록 리스트 초기화됨")
    
    def print_decided_blocks(self):
        """결정된 블록들의 상세 정보 출력"""
        if not self._decided_blocks:
            print("⚠️  결정된 블록이 없습니다")
            return
        
        print("\n" + "*" * 30)
        print(f"📋 결정된 블록 목록 (총 {len(self._decided_blocks)}개)")
        print("*" * 30)
        
        for block in self._decided_blocks:
            print(f"\n[{block.click_order}] {block}")
            if block.is_valid and block.center_3d_mm:
                x, y, z = block.center_3d_mm
                print(f"    → 위치: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                print(f"    → 크기: {block.side_length_mm:.1f} mm")
        
        print("\n" + "*" * 30 + "\n")
    
    # -------------------- 편의 메서드 --------------------
    
    def get_closest_block(self, update: bool = True) -> Optional[Block]:
        """가장 가까운 블록 반환"""
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return min(blocks, key=lambda b: b.depth)
    
    def get_farthest_block(self, update: bool = True) -> Optional[Block]:
        """가장 먼 블록 반환"""
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.depth)
    
    def get_largest_block(self, update: bool = True) -> Optional[Block]:
        """가장 큰 블록 반환"""
        blocks = self.get_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.area)
    
    def get_block_count(self, update: bool = True) -> int:
        """감지된 블록 수"""
        return len(self.get_blocks(update))
    
    def find_blocks_in_depth_range(self,
                                   min_depth: float = 0,
                                   max_depth: float = float('inf'),
                                   update: bool = True) -> List[Block]:
        """특정 깊이 범위의 블록들 반환"""
        blocks = self.get_valid_blocks(update)
        return [b for b in blocks if min_depth <= b.depth <= max_depth]
    
    def find_block_at(self, x: int, y: int, 
                      update: bool = False) -> Optional[Block]:
        """특정 픽셀 위치의 블록 반환"""
        blocks = self.get_blocks(update)
        for block in blocks:
            bx, by, bw, bh = block.bbox
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return block
        return None
    
    @property
    def last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame
    
    @property
    def last_depth(self) -> Optional[np.ndarray]:
        return self._last_depth
    
    @property
    def last_blocks(self) -> List[Block]:
        return self._last_blocks.copy()
    
    def get_depth_at(self, x: int, y: int) -> float:
        """특정 픽셀의 깊이값 (미터)"""
        if self._last_depth is None:
            return 0.0
        return self._camera.get_depth_at(x, y, self._last_depth)
    
    def get_3d_at(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """특정 픽셀의 3D 좌표 (미터)"""
        if self._last_depth is None:
            return None
        return self._camera.pixel_to_3d(x, y, self._last_depth)
    
    # -------------------- 디버그 GUI --------------------
    
    def run_debug(self):
        """
        디버그 GUI 실행
        
        조작:
            - 블록 클릭: 임시 리스트에 추가
            - 빈 공간 클릭: 깊이 확인
            - 's' 키: 클릭한 블록들을 결정 상태로 확정 (덮어쓰기)
            - 'p' 키: 결정된 블록 출력
            - 'c' 키: 클릭한 블록 리스트만 초기화
            - 'r' 키: 모든 블록 리스트 초기화
            - ESC: 종료
        """
        print("\n" + "=" * 50)
        print("🏗️ Block Detection - Debug Mode v6.2")
        print("=" * 50)
        print("📌 조작:")
        print("   - 블록 클릭: 임시 리스트에 추가")
        print("   - 빈 공간 클릭: 깊이 확인")
        print("   - 's' 키: 결정 (클릭 → 결정, 덮어쓰기)")
        print("   - 'p' 키: 결정된 블록 출력")
        print("   - 'c' 키: 클릭 리스트만 초기화")
        print("   - 'r' 키: 전체 초기화")
        print("   - ESC: 종료")
        print("=" * 50 + "\n")
        
        selected_idx = -1
        
        def on_mouse(event, x, y, flags, param):
            nonlocal selected_idx
            
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            
            # 블록 클릭 확인
            for i, block in enumerate(self._last_blocks):
                bx, by, bw, bh = block.bbox
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    selected_idx = i
                    
                    block_copy = Block(
                        bbox=block.bbox,
                        center_2d=block.center_2d,
                        contour=block.contour,
                        rotated_box=block.rotated_box,
                        area=block.area,
                        aspect_ratio=block.aspect_ratio,
                        solidity=block.solidity,
                        angle=block.angle,
                        center_3d=block.center_3d,
                        depth=block.depth,
                        real_width_mm=block.real_width_mm,
                        real_height_mm=block.real_height_mm,
                        click_order=len(self._clicked_blocks) + 1
                    )
                    
                    self._clicked_blocks.append(block_copy)
                    self._print_block_info(block_copy)
                    return
            
            # 빈 공간 클릭
            selected_idx = -1
            self._print_depth_info(x, y)
        
        cv2.namedWindow("Result")
        cv2.setMouseCallback("Result", on_mouse)
        
        cv2.namedWindow("Control")
        cv2.createTrackbar("Threshold", "Control", 
                          self.config.threshold, 255, lambda x: None)
        cv2.createTrackbar("Min Area", "Control", 
                          self.config.min_area, 5000, lambda x: None)
        cv2.createTrackbar("Max Area", "Control", 
                          self.config.max_area, 30000, lambda x: None)
        
        try:
            while True:
                self.config.threshold = cv2.getTrackbarPos("Threshold", "Control")
                self.config.min_area = cv2.getTrackbarPos("Min Area", "Control")
                self.config.max_area = cv2.getTrackbarPos("Max Area", "Control")
                
                if not self.update():
                    continue
                
                display = self._draw_result(selected_idx)
                cv2.imshow("Result", display)
                
                if self._detector.binary_view is not None:
                    cv2.imshow("Binary (ROI)", self._detector.binary_view)
                
                depth_display = self._draw_depth()
                cv2.imshow("Depth", depth_display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # 결정
                    self.decide_blocks()
                elif key == ord('p'):  # 결정 블록 출력
                    self.print_decided_blocks()
                elif key == ord('c'):  # 클릭 리스트만 초기화
                    self.clear_clicked_blocks()
                elif key == ord('r'):  # 전체 초기화
                    self.reset_all()
                    selected_idx = -1
        
        finally:
            cv2.destroyAllWindows()
    
    def _draw_result(self, selected_idx: int = -1) -> np.ndarray:
        """결과 화면 그리기"""
        if self._last_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        display = self._last_frame.copy()
        
        # ROI 영역 표시
        cfg = self.config
        cv2.rectangle(display, 
                     (cfg.roi_x, cfg.roi_y),
                     (cfg.roi_x + cfg.roi_w, cfg.roi_y + cfg.roi_h),
                     (255, 255, 0), 2)
        
        # 감지된 블록 그리기
        for i, block in enumerate(self._last_blocks):
            color = (0, 255, 0) if i == selected_idx else (0, 165, 255)
            
            # 회전 박스
            cv2.drawContours(display, [block.rotated_box], 0, color, 2)
            
            # 중심점
            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 5, color, -1)
            
            # 깊이 정보
            if block.is_valid:
                text = f"{block.depth_cm:.1f}cm"
                cv2.putText(display, text, (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 클릭한 블록 표시
        for block in self._clicked_blocks:
            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 8, (255, 0, 255), 2)
            cv2.putText(display, str(block.click_order), (cx - 5, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 결정된 블록 표시
        for block in self._decided_blocks:
            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 10, (0, 255, 255), 3)
            cv2.putText(display, f"D{block.click_order}", (cx - 10, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 상태 정보 표시
        info_y = 30
        cv2.putText(display, f"Detected: {len(self._last_blocks)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(display, f"Clicked: {len(self._clicked_blocks)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        info_y += 25
        cv2.putText(display, f"Decided: {len(self._decided_blocks)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display
    
    def _draw_depth(self) -> np.ndarray:
        """깊이 맵 시각화"""
        if self._last_depth is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self._last_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        return depth_colormap
    
    def _print_block_info(self, block: Block):
        """블록 정보 콘솔 출력"""
        print(f"\n{'='*50}")
        print(f"🎯 블록 #{block.click_order} 선택됨")
        print(f"{'='*50}")
        print(f"📍 2D 중심: {block.center_2d}")
        
        if block.is_valid and block.center_3d_mm:
            x, y, z = block.center_3d_mm
            print(f"📍 3D 좌표: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
            print(f"📏 깊이: {block.depth_cm:.1f} cm")
            print(f"📐 크기: {block.side_length_mm:.1f} mm")
        else:
            print("⚠️  깊이 정보 없음")
        
        print(f"{'='*50}\n")
    
    def _print_depth_info(self, x: int, y: int):
        """깊이 정보 콘솔 출력"""
        depth = self.get_depth_at(x, y)
        
        print(f"\n{'='*50}")
        print(f"📍 픽셀 ({x}, {y}) 클릭")
        print(f"{'='*50}")
        
        if depth > 0:
            print(f"📏 깊이: {depth * 100:.1f} cm ({depth * 1000:.1f} mm)")
            point_3d = self.get_3d_at(x, y)
            if point_3d:
                px, py, pz = point_3d
                print(f"📍 3D 좌표: ({px*1000:.1f}, {py*1000:.1f}, {pz*1000:.1f}) mm")
        else:
            print("⚠️  깊이 정보 없음")
        
        print(f"{'='*50}\n")


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    with BlockDetectionSystem() as system:
        system.run_debug()
