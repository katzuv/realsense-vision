import os
from ultralytics import YOLO

import app.core.logging_config as logging_config
from app.config import ConfigManager
from app.constants import CHIP_TYPE_QCS6490

logger = logging_config.get_logger(__name__)


class YOLODetector:
    def __init__(self, model_path, imgsz=640):
        logger.info(
            f"Initializing YOLO detector with model: {model_path}", operation="init"
        )

        config = ConfigManager().get()
        chip_type = config.rknn_chip_type
        
        # Log chip-specific initialization
        if chip_type == CHIP_TYPE_QCS6490:
            logger.info("Initializing for QCS6490 with ONNX Runtime (QNN provider)", operation="init")
        else:
            logger.info(f"Initializing for {chip_type} with RKNN Runtime", operation="init")
        
        # Initialize YOLO model with chip-specific settings
        if chip_type == CHIP_TYPE_QCS6490:
            # For QCS6490, configure ONNX Runtime to use QNN execution provider for NPU acceleration
            try:
                import onnxruntime as ort
                # Set environment variable to enable QNN execution provider
                # This will be used by ultralytics YOLO when loading ONNX models
                os.environ["ORT_EXECUTION_PROVIDERS"] = "QNNExecutionProvider,CPUExecutionProvider"
                
                # Log available providers
                available_providers = ort.get_available_providers()
                logger.info(f"ONNX Runtime available providers: {available_providers}", operation="init")
                
                if "QNNExecutionProvider" in available_providers:
                    logger.info("QNN execution provider is available for NPU acceleration", operation="init")
                else:
                    logger.warning("QNN execution provider not available, will use CPU", operation="init")
            except ImportError:
                logger.warning("onnxruntime not available, using default providers", operation="init")
            except Exception as e:
                logger.warning(f"Failed to configure QNN provider: {e}, using default providers", operation="init")
        
        # Initialize YOLO model (supports both RKNN and ONNX formats)
        self.model = YOLO(model_path, task="detect")
        self.imgsz = imgsz
        self.results = None
        self.detection_count = 0
        self.chip_type = chip_type

        logger.info(
            f"YOLO detector initialized successfully (chip={chip_type}, imgsz={imgsz})",
            operation="init",
            status="success",
        )

    def detect(self, image):
        """Run detection on an image"""
        if image is None:
            logger.warning("Received None image for detection", operation="detect")
            return

        self.results = self.model(
            image, imgsz=self.imgsz, conf=ConfigManager().get().min_confidence
        )[0]

        self.detection_count += 1

        # Log periodically
        if self.detection_count % 100 == 0:
            logger.debug(
                f"Processed {self.detection_count} detections", operation="detect"
            )

    def get_annotated_image(self):
        """Get annotated image with detections."""
        try:
            if self.results is None:
                return None
            return self.results.plot()
        except Exception as e:
            logger.error(
                f"Error plotting annotations: {e}", operation="get_annotated_image"
            )
            return None

    def get_detections(self):
        """Get detection results as numpy arrays."""
        try:
            if self.results is None:
                logger.warning("No results available", operation="get_detections")
                return None

            return (
                self.results.boxes.xyxy.cpu().numpy(),
                self.results.boxes.conf.cpu().numpy(),
                self.results.boxes.cls.cpu().numpy(),
            )
        except Exception as e:
            logger.error(
                f"Error extracting detections: {e}", operation="get_detections"
            )
            return None
