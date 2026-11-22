"""
TrafficVision - Input Validation Utilities
Validates uploaded files, checks MIME types, verifies sizes, and sanitizes filenames.
"""

import os
from pathlib import Path
import uuid
from fastapi import UploadFile, HTTPException
from app.config import settings


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"BM": "image/bmp",
    b"RIFF": "image/webp"
}


def sanitize_filename(filename: str) -> str:
    """
    Prevents path traversal and injects a unique UUID prefix.
    """
    clean_stem = Path(filename).name
    ext = Path(clean_stem).suffix.lower()
    unique_id = uuid.uuid4().hex[:8]
    sanitized_stem = "".join(c for c in Path(clean_stem).stem if c.isalnum() or c in ("-", "_"))
    return f"{sanitized_stem}_{unique_id}{ext}"


async def validate_image_upload(file: UploadFile) -> bytes:
    """
    Validates file extension, upload size, and byte contents.
    Returns the read bytes.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Read contents with size check
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    # Magic signature verification
    is_valid_header = any(content.startswith(sig) for sig in MAGIC_SIGNATURES)
    if not is_valid_header:
        raise HTTPException(
            status_code=400,
            detail="File content does not match a valid image signature."
        )

    return content
