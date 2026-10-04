"""Pinned PaddleX batch configuration and lightweight runtime observations."""
from contextlib import contextmanager
from copy import deepcopy
import time


BATCH_PATHS = {
    "pages": ("batch_size",),
    "layout": ("SubModules", "LayoutDetection", "batch_size"),
    "text_recognition": ("SubPipelines", "GeneralOCR", "SubModules", "TextRecognition", "batch_size"),
    "textline_orientation": ("SubPipelines", "GeneralOCR", "SubModules", "TextLineOrientation", "batch_size"),
    "formula": ("SubPipelines", "FormulaRecognition", "SubModules", "FormulaRecognition", "batch_size"),
}


def scaled_config(config, scale):
    """Copy the complete official config: partial dicts replace, not merge."""
    result = deepcopy(config)
    for path in BATCH_PATHS.values():
        node = result
        for key in path[:-1]:
            node = node[key]
        value = node[path[-1]]
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid official batch size at {path}: {value!r}")
        node[path[-1]] = value * scale
    return result


def leaf_models(native):
    return {
        "layout": native.layout_det_model,
        "text_recognition": native.general_ocr_pipeline.text_rec_model,
        "textline_orientation": native.general_ocr_pipeline.textline_orientation_model,
        "formula": native.formula_recognition_pipeline.formula_recognition_model,
    }


def prepare_runtime(native):
    # These two components otherwise load on the first page, even with supplied
    # overall OCR. Same factories/configs as TableRecognitionPipelineV2.predict;
    # no inference, dummy input, or warmup. The adapter keeps table orientation on.
    table = native.table_recognition_pipeline._pipeline
    if table.table_orientation_classify_model is None:
        table.table_orientation_classify_model = table.create_model(table.table_orientation_classify_config)
    if table.general_ocr_pipeline is None:
        table.general_ocr_pipeline = table.create_pipeline(table.general_ocr_config_bak)
    return {"pages": native.batch_sampler.batch_size,
            **{name: model.batch_sampler.batch_size for name, model in leaf_models(native).items()}}


class TimedStage:
    """Count active generator time, excluding time spent in its consumer."""
    def __init__(self, stage, stats):
        self.stage, self.stats = stage, stats

    def __getattr__(self, name):
        return getattr(self.stage, name)

    def __call__(self, *args, **kwargs):
        self.stats["calls"] += 1
        iterator = None
        started = time.perf_counter()
        try:
            iterator = iter(self.stage(*args, **kwargs))
        finally:
            self.stats["active_seconds"] += time.perf_counter() - started
        try:
            while True:
                started = time.perf_counter()
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                finally:
                    self.stats["active_seconds"] += time.perf_counter() - started
                yield item
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


@contextmanager
def observe_runtime(native):
    """Observe this pipeline only; restore all objects/methods on exit."""
    stats = {"stages": {}, "microbatches": {}}
    originals = []
    try:
        for name, model in leaf_models(native).items():
            record = {"sizes": {}, "failed_calls": 0}
            stats["microbatches"][name] = record
            original = model.process

            def process(batch_data, _original=original, _record=record, **kwargs):
                size = str(len(batch_data.instances))
                _record["sizes"][size] = _record["sizes"].get(size, 0) + 1
                try:
                    return _original(batch_data, **kwargs)
                except Exception:
                    _record["failed_calls"] += 1
                    raise

            originals.append((model, "process", original))
            model.process = process
        for name in ("layout_det_model", "region_detection_model", "formula_recognition_pipeline",
                     "general_ocr_pipeline", "table_recognition_pipeline"):
            stage = getattr(native, name, None)  # Region detection is optional.
            if stage is not None:
                record = {"calls": 0, "active_seconds": 0.0}
                stats["stages"][name] = record
                originals.append((native, name, stage))
                setattr(native, name, TimedStage(stage, record))
        yield stats
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
