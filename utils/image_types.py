import base64
import mimetypes
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedImage:
    """A single image extracted from a document or loaded from a file."""
    data: bytes
    media_type: str
    source_page: Optional[int] = None
    label: str = ""

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclass
class ExtractedContent:
    """Text + optional images extracted from a file."""
    text: str
    images: list[ExtractedImage] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0
