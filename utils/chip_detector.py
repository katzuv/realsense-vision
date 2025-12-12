"""
Utility module for auto-detecting the chip/platform type.

Supports detection of:
- Rockchip: RK3588, RK3588s, RK3576, RK3568
- Qualcomm: QCS6490
"""

import os
import logging

logger = logging.getLogger(__name__)


def detect_chip_type() -> str:
    """
    Auto-detect the chip type based on system information.
    
    Returns:
        str: Detected chip type identifier (e.g., 'rk3588', 'qcs6490')
             Defaults to 'rk3588' if detection fails.
    """
    
    # Method 1: Check device tree model/compatible files
    # These are standard locations on ARM SBCs running Linux
    device_tree_paths = [
        "/proc/device-tree/model",
        "/proc/device-tree/compatible",
        "/sys/firmware/devicetree/base/model",
        "/sys/firmware/devicetree/base/compatible",
    ]
    
    for path in device_tree_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read content and convert to lowercase for case-insensitive matching
                    content = f.read().lower().replace('\x00', ' ')
                    
                    # Check for Qualcomm QCS6490 first (most specific)
                    if 'qcs6490' in content:
                        logger.info(f"Detected QCS6490 from {path}")
                        return 'qcs6490'
                    
                    # Check for Rockchip variants (order matters - check more specific first)
                    if 'rk3588s' in content:
                        logger.info(f"Detected RK3588s from {path}")
                        return 'rk3588s'
                    if 'rk3588' in content:
                        logger.info(f"Detected RK3588 from {path}")
                        return 'rk3588'
                    if 'rk3576' in content:
                        logger.info(f"Detected RK3576 from {path}")
                        return 'rk3576'
                    if 'rk3568' in content:
                        logger.info(f"Detected RK3568 from {path}")
                        return 'rk3568'
                    
                    # Broader Qualcomm detection
                    if 'qualcomm' in content or 'qcom' in content:
                        logger.info(f"Detected Qualcomm chip from {path}, defaulting to QCS6490")
                        return 'qcs6490'
        except (IOError, OSError) as e:
            logger.debug(f"Could not read {path}: {e}")
            continue
    
    # Method 2: Check cpuinfo as fallback
    try:
        with open('/proc/cpuinfo', 'r', encoding='utf-8') as f:
            cpuinfo = f.read().lower()
            if 'qualcomm' in cpuinfo or 'qcom' in cpuinfo:
                logger.info("Detected Qualcomm chip from /proc/cpuinfo, defaulting to QCS6490")
                return 'qcs6490'
            if 'rockchip' in cpuinfo:
                logger.info("Detected Rockchip chip from /proc/cpuinfo, defaulting to RK3588")
                return 'rk3588'
    except (IOError, OSError) as e:
        logger.debug(f"Could not read /proc/cpuinfo: {e}")
    
    # Default fallback
    logger.warning("Could not auto-detect chip type, defaulting to RK3588")
    return 'rk3588'


def resolve_chip_type(config_value: str) -> str:
    """
    Resolve the chip type from configuration value.
    
    If the config value is "auto", performs auto-detection.
    Otherwise, returns the config value as-is.
    
    Args:
        config_value: Chip type from configuration
        
    Returns:
        str: Resolved chip type identifier
    """
    if config_value.lower() == "auto":
        detected = detect_chip_type()
        logger.info(f"Auto-detected chip type: {detected}")
        return detected
    return config_value
