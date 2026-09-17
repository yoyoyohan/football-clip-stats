from ultralytics import YOLO
import supervision as sv

from utils.class_filter import (
    allowed_class_ids,
    filter_supervision_detections,
    split_trackable_overlay,
)
from utils.detection_utils import detection_from_sv, extract_ball
from utils.goal_regions import detection_from_raw, extract_overlay


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size], conf=0.1, verbose=False
            )
            detections += detections_batch
        return detections

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
