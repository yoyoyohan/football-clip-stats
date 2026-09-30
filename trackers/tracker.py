from ultralytics import YOLO
import supervision as sv

from utils.class_filter import (
    allowed_class_ids,
    filter_supervision_detections,
    split_trackable_overlay,
)
from utils.detection_utils import detection_from_sv, extract_ball, extract_coco_sports_ball
from utils.goal_regions import detection_from_raw, extract_overlay


class Tracker:
    def __init__(
        self,
        model_path,
        fps: float = 30.0,
        device=None,
        ball_fallback: bool = True,
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.ball_fallback = ball_fallback
        self._coco_ball_model = None
        frame_rate = max(1, int(round(fps)))
        # Keep ByteTrack; only scale the lost-track buffer to real video fps
        # so ~1s of occlusion does not immediately drop an identity.
        self.tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            lost_track_buffer=max(30, frame_rate),
        )

    def _coco(self) -> YOLO:
        if self._coco_ball_model is None:
            self._coco_ball_model = YOLO("yolov8n.pt")
        return self._coco_ball_model

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        predict_kw = {"conf": 0.1, "verbose": False}
        if self.device is not None:
            predict_kw["device"] = self.device
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], **predict_kw)
            detections += detections_batch
        return detections

    def _fallback_ball(self, frame):
        if not self.ball_fallback:
            return None
        kw = {"conf": 0.15, "verbose": False, "classes": [32]}
        if self.device is not None:
            kw["device"] = self.device
        try:
            result = self._coco().predict(frame, **kw)[0]
        except Exception:
            return None
        return extract_coco_sports_ball(result)

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        frame_records = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_supervision = filter_supervision_detections(
                detection_supervision, allowed_class_ids(cls_names)
            )
            trackable, overlay = split_trackable_overlay(detection_supervision, cls_names)

            if "goalkeeper" in cls_names_inv and "player" in cls_names_inv:
                gk_id = cls_names_inv["goalkeeper"]
                player_id = cls_names_inv["player"]
                class_ids = trackable.class_id.copy()
                class_ids[class_ids == gk_id] = player_id
                trackable = sv.Detections(
                    xyxy=trackable.xyxy,
                    mask=trackable.mask,
                    confidence=trackable.confidence,
                    class_id=class_ids,
                    tracker_id=trackable.tracker_id,
                    data=trackable.data,
                )
            # Ball from pre-track detections: ByteTrack often drops intermittent
            # small-object hits, which zeroed ball recall with newer weights.
            ball = extract_ball(trackable, cls_names_inv)
            if ball is None:
                ball = self._fallback_ball(frames[frame_num])

            detection_with_tracks = self.tracker.update_with_detections(trackable)

            frame_detections = []
            for i in range(len(detection_with_tracks)):
                det_dict = detection_from_sv(detection_with_tracks, i, cls_names)
                if det_dict is not None:
                    frame_detections.append(det_dict)
            for i in range(len(overlay)):
                det_dict = detection_from_raw(overlay, i, cls_names, require_track_id=False)
                if det_dict is not None:
                    frame_detections.append(det_dict)
            frame_records.append(
                {
                    "frame_idx": frame_num,
                    "detections": frame_detections,
                    "ball": ball,
                    "overlay": extract_overlay(frame_detections),
                    "frame": frames[frame_num],
                }
            )

        return frame_records
