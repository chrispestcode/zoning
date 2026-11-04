import base64
import io
import PIL
import PIL.Image as Image
from loguru import logger
from typing import Optional

def encode_image(image: Image.Image, format: str = "JPEG", max_size: tuple = (800, 800)) -> Optional[str]:
    """Encode image to base64 with proper buffer handling"""
    try:
        # Handle transparency if saving as JPEG
        if format.upper() == "JPEG":
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
        
        # Resize the image if larger than max_size
        if max_size:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Create a new BytesIO buffer for each save operation
        buffer = io.BytesIO()
        save_kwargs = {"format": format.upper()}
        if format.upper() == "JPEG":
            save_kwargs.update({"quality": 85, "optimize": True})
        image.save(buffer, **save_kwargs)
        
        # Get the encoded string
        buffer.seek(0)
        encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Close the buffer
        buffer.close()
        
        return f"data:image/{format.lower()};base64,{encoding}"
    
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return ""
    finally:
        # Ensure the image is closed
        if hasattr(image, 'close'):
            image.close()

def decode_image(base64_str: str):
    base64_str = clean_picture_base64_prefix(base64_str)
    try:
        return Image.open(io.BytesIO(base64.b64decode(base64_str)))
    except Exception:
        logger.error("Picture could not be decoded.")

def clean_picture_base64_prefix(base64_uri):
        start_idx = base64_uri.find('base64,')
        base64_str = base64_uri[start_idx+len('base64,'):]
        return base64_str

def transpose_image(image: Image.Image) -> Image.Image:
    """Process image with memory optimization"""
    try:
        image = PIL.ImageOps.exif_transpose(image) or image
        return image.convert("RGB")
    except Exception as e:
        print(f"Error transposing image: {e}")
        return image

def get_bounding_box(table):
    l, t, r, b = [], [], [], []
    table_cells = table.data.table_cells
    for cell in table_cells:
        l.append(cell.bbox.l)
        t.append(cell.bbox.t)
        r.append(cell.bbox.r)
        b.append(cell.bbox.b)
    return {
            'left' : min(l),
            'top': min(t),
            'right': max(r),
            'bottom': max(b)
        }