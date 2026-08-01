from training.soccernet_to_yolo import bbox_to_yolo_line, line_to_yolo_line

IMAGE_META = {"width": 1920, "height": 1080}


def test_bbox_to_yolo_line_maps_player():
    line = bbox_to_yolo_line(
        {
            "class": "Player team left",
            "points": {"x1": 100, "y1": 200, "x2": 300, "y2": 500},
        },
        IMAGE_META,
    )
    assert line is not None
    assert line.startswith("1 ")


def test_line_to_yolo_line_maps_goalpost():
    line = line_to_yolo_line(
        {
            "class": "Goal left post left ",
            "points": [100.0, 200.0, 110.0, 400.0],
        },
        IMAGE_META,
    )
    assert line is not None
    assert line.startswith("4 ")
