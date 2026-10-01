import logging
from typing import Dict, List, Optional, Any
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

def _convert_to_degrees(value) -> Optional[float]:
    """Helper function to convert the GPS coordinates stored in the EXIF to degrees in float format"""
    try:
        if not value:
            return None
        d, m, s = value
        return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)
    except Exception as e:
        logger.debug(f"Error converting GPS: {e}")
        return None

def extract_exif(image_path: str) -> Dict[str, Any]:
    """
    Extracts EXIF metadata from an image.
    Returns:
        Dict containing taken_at, camera_model, gps_lat, gps_lon, width, height, orientation.
    """
    exif_data = {
        'taken_at': None,
        'camera_model': None,
        'gps_lat': None,
        'gps_lon': None,
        'width': None,
        'height': None,
        'orientation': None
    }
    
    try:
        with Image.open(image_path) as img:
            exif_data['width'] = img.width
            exif_data['height'] = img.height
            
            exif = img._getexif()
            if not exif:
                return exif_data
                
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                
                if tag == 'DateTimeOriginal':
                    exif_data['taken_at'] = value
                elif tag == 'Model':
                    exif_data['camera_model'] = str(value).strip()
                elif tag == 'Orientation':
                    exif_data['orientation'] = value
                elif tag == 'GPSInfo':
                    gps_info = {}
                    for t in value:
                        sub_tag = ExifTags.GPSTAGS.get(t, t)
                        gps_info[sub_tag] = value[t]
                        
                    lat = gps_info.get('GPSLatitude')
                    lat_ref = gps_info.get('GPSLatitudeRef')
                    lon = gps_info.get('GPSLongitude')
                    lon_ref = gps_info.get('GPSLongitudeRef')
                    
                    if lat and lat_ref and lon and lon_ref:
                        lat_val = _convert_to_degrees(lat)
                        lon_val = _convert_to_degrees(lon)
                        if lat_val is not None and lon_val is not None:
                            if lat_ref != 'N':
                                lat_val = -lat_val
                            if lon_ref != 'E':
                                lon_val = -lon_val
                            exif_data['gps_lat'] = lat_val
                            exif_data['gps_lon'] = lon_val
    except Exception as e:
        logger.warning(f"Failed to extract EXIF from {image_path}: {e}")
        
    return exif_data

def extract_exif_batch(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Extracts EXIF data for a batch of images."""
    return [extract_exif(path) for path in image_paths]
