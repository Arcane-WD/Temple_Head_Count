from ultralytics import solutions
import config

class TempleCounter:
    def __init__(self):
        self.counter = solutions.ObjectCounter(
            model=config.MODEL_PATH,
            region=config.GATE_LINE,
            classes=config.TARGET_CLASSES,
            conf=config.CONF_THRESH,
            iou=config.IOU_THRESH,
            tracker=config.TRACKER_CONFIG,
            show=False,
            verbose=False
        )

        if config.DEVICE == "cuda":
            self.counter.model.to(config.DEVICE)

    def process_frame(self, frame):
        results = self.counter(frame)

        if isinstance(results, list):
            annotated_frame = results[0].plot_im
        else:
            annotated_frame = results.plot_im
        return annotated_frame, self.counter.in_count, self.counter.out_count