import asyncio
import os
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from app.config import ConfigManager
from app.constants import CHIP_TYPE_QCS6490
from app.core import logging_config
from convert_model import async_convert_model

UPLOAD_FOLDER = Path("uploads")
logger = logging_config.get_logger(__name__)


def secure_filename(name: str) -> str:
    # very simple sanitization
    return os.path.basename(name).replace("..", "").replace("/", "").replace("\\", "")


async def upload_model(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No selected file")

        raw_name = secure_filename(file.filename)
        base = Path(raw_name).stem
        file_path = UPLOAD_FOLDER / f"{base}.pt"

        UPLOAD_FOLDER.mkdir(exist_ok=True)

        if file_path.exists():
            file_path.unlink()

        file_path.write_bytes(await file.read())
        logger.info(f"Finished uploading {file_path}", operation="upload")

        await asyncio.sleep(1)

        chip = ConfigManager().get().rknn_chip_type
        asyncio.create_task(async_convert_model(file_path, chip))

        return RedirectResponse(url="/upload_progress", status_code=303)

    except Exception as e:
        logger.exception("Model conversion failed", operation="upload")
        raise HTTPException(status_code=500, detail=f"Error converting model: {e}")


def get_all_rknn_models() -> list[str]:
    """Get all converted models (RKNN or ONNX depending on chip type)."""
    try:
        if not UPLOAD_FOLDER.exists():
            return []

        chip_type = ConfigManager().get().rknn_chip_type
        
        models = []
        for path in UPLOAD_FOLDER.iterdir():
            if not path.is_dir():
                continue
            
            name = path.name
            
            # Accept both RKNN and ONNX model directories based on chip type
            if chip_type == CHIP_TYPE_QCS6490:
                # For QCS6490, look for ONNX model directories
                if any(keyword in name.lower() for keyword in ("onnx", CHIP_TYPE_QCS6490)) and name.endswith("_model"):
                    models.append(name)
            elif name.endswith("_rknn_model"):
                # For Rockchip chips, look for RKNN model directories
                models.append(name)
        
        return models

    except Exception:
        logger.exception("Failed to list models", operation="list_models")
        return []
