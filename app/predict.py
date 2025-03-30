from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

def simple_predict(
    image_path,
    model_path,
    model_type='yolov8',
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    conf_threshold=0.25
):
    detection_model = AutoDetectionModel.from_pretrained(
        model_path=model_path,
        model_type=model_type,
        confidence_threshold=conf_threshold
    )
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio
    )
    return result

def extract_sorted_text_with_newlines(result, ignore_labels=None, line_threshold=10):
    if ignore_labels is None:
        ignore_labels = []

    boxes = [
        {
            "bbox": obj.bbox.to_xyxy(),
            "label": obj.category.name
        }
        for obj in result.object_prediction_list
        if obj.category.name not in ignore_labels
    ]

    sorted_boxes = sorted(boxes, key=lambda x: (x["bbox"][1], x["bbox"][0]))

    lines = []
    current_line = []
    last_y = None

    for box in sorted_boxes:
        y_min = box["bbox"][1]
        if last_y is None or abs(y_min - last_y) > line_threshold:
            if current_line:
                lines.append(current_line)
            current_line = [box]
        else:
            current_line.append(box)
        last_y = y_min

    if current_line:
        lines.append(current_line)

    sorted_text = "\n".join(
        "".join(box["label"] for box in sorted(line, key=lambda x: x["bbox"][0]))
        for line in lines
    )

    return sorted_text
