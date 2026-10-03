import os
import sys
import io
import requests
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import ee

def get_decimal_from_dms(dms, ref):
    """Convert GPS coordinates from DMS to decimal format."""
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_exif_data(image_path):
    """Extract GPS and date information from image EXIF data."""
    img = Image.open(image_path)
    exif = img._getexif()
    if not exif:
        return None, None
        
    # Find GPS info
    gps_info = {}
    date_taken = None
    from PIL.ExifTags import TAGS, GPSTAGS
    for tag, value in exif.items():
        decoded = TAGS.get(tag, tag)
        if decoded == 'GPSInfo':
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_info[sub_decoded] = value[t]
        elif decoded == 'DateTimeOriginal':
            date_taken = value

    if not gps_info or 'GPSLatitude' not in gps_info or 'GPSLongitude' not in gps_info:
        return None, date_taken

    lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
    lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
    
    # Parse date
    capture_date = None
    if date_taken:
        try:
            capture_date = datetime.strptime(date_taken, '%Y:%m:%d %H:%M:%S').date()
        except:
            pass
            
    return (lat, lon), capture_date

def get_sentinel_image(geom, start_date, end_date):
    """Fetch Sentinel-2 imagery from Google Earth Engine."""
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))  # Relaxed cloud cover threshold to 80%

    # Debugging: Check the size of the image collection
    collection_size = collection.size().getInfo()
    print(f"Image collection size for {start_date} to {end_date}: {collection_size}")

    if collection_size == 0:
        return None
    return collection.median().clip(geom)

def compute_area(image, geom, index_type):
    """Compute area of vegetation or water using NDVI/NDWI."""
    if index_type == "NDVI":
        index_img = image.normalizedDifference(['B8', 'B4'])
        mask = index_img.gt(0.4)  # Dense vegetation
    else:
        index_img = image.normalizedDifference(['B3', 'B8'])
        mask = index_img.gt(0.0)  # Water
        
    area_img = ee.Image.pixelArea().updateMask(mask)
    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    return stats.get('area', 0)

def generate_infographic(input_image_path):
    """Generate an infographic comparing pre- and post-event satellite imagery."""
    # Hardcode your Earth Engine Project ID here
    EE_PROJECT_ID = "beeren-gemni-integration"  # Replace with your actual project ID

    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT_ID)

    print("Extracting EXIF data...")
    coords, capture_date = extract_exif_data(input_image_path)
    if not coords:
        print("Error: No GPS data found in the image EXIF! Make sure you pass an original JPEG from a camera/phone with location services enabled.")
        sys.exit(1)
        
    lat, lon = coords
    if not capture_date:
        capture_date = datetime.now().date()
        
    print(f"Location found: {lat:.4f}, {lon:.4f}")
    print(f"Capture date: {capture_date}")
    
    # Create bounding box (+/- 0.05 degrees is ~5km)
    buffer = 0.05
    geom = ee.Geometry.Rectangle([lon - buffer, lat - buffer, lon + buffer, lat + buffer])
    
    # Extend the date range to increase the likelihood of finding imagery
    pre_start = (capture_date - timedelta(days=120)).strftime('%Y-%m-%d')  # Extended to 120 days before
    pre_end = capture_date.strftime('%Y-%m-%d')
    post_start = capture_date.strftime('%Y-%m-%d')
    post_end = (capture_date + timedelta(days=365)).strftime('%Y-%m-%d')  # Extended to 1 year after

    print("Fetching Earth Engine imagery...")
    pre_image = get_sentinel_image(geom, pre_start, pre_end)
    post_image = get_sentinel_image(geom, post_start, post_end)
    
    if not pre_image:
        print("Error: Could not find pre-event satellite imagery for those dates.")
        sys.exit(1)

    if not post_image:
        print("Warning: Could not find post-event satellite imagery for those dates. Proceeding with pre-event imagery only.")
        post_image = pre_image  # Use pre-event imagery as a fallback

    print("Calculating statistics...")
    pre_veg = compute_area(pre_image, geom, "NDVI")
    post_veg = compute_area(post_image, geom, "NDVI")
    pre_water = compute_area(pre_image, geom, "NDWI")
    post_water = compute_area(post_image, geom, "NDWI")
    
    veg_diff = post_veg - pre_veg
    water_diff = post_water - pre_water

    print("Rendering images...")
    vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
    
    # Create cyan dashed outline
    fc = ee.FeatureCollection([ee.Feature(geom)])
    empty = ee.Image().byte()
    outline = empty.paint(fc, 1, 3)
    outline_rgb = ee.Image.rgb(0, 255, 255).updateMask(outline.neq(0))
    
    pre_painted = pre_image.visualize(**vis_params).blend(outline_rgb)
    post_painted = post_image.visualize(**vis_params).blend(outline_rgb)
    
    pre_url = pre_painted.getThumbURL({'dimensions': 600, 'region': geom, 'format': 'png'})
    post_url = post_painted.getThumbURL({'dimensions': 600, 'region': geom, 'format': 'png'})
    
    pre_img = Image.open(io.BytesIO(requests.get(pre_url).content)).convert("RGBA")
    post_img = Image.open(io.BytesIO(requests.get(post_url).content)).convert("RGBA")
    
    # Composite (1200x600 for images, +150px for stats text at the bottom = 1200x750)
    composite = Image.new('RGBA', (1200, 750), (30, 30, 30, 255))
    composite.paste(pre_img, (0, 0))
    composite.paste(post_img, (600, 0))
    
    draw = ImageDraw.Draw(composite)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    def draw_text(x, y, text, font, fill="white", outline="black"):
        for adj in [-2, -1, 1, 2]:
            for adj2 in [-2, -1, 1, 2]:
                draw.text((x+adj, y+adj2), text, font=font, fill=outline)
        draw.text((x, y), text, font=font, fill=fill)

    # Labels
    draw_text(180, 500, "BEFORE", font_large)
    draw_text(780, 500, "AFTER", font_large)
    
    # Stats Text
    sign_veg = "+" if veg_diff > 0 else ""
    sign_water = "+" if water_diff > 0 else ""
    
    stats_text = (
        f"Location: {lat:.4f}, {lon:.4f}  |  Event Date: {capture_date}\n\n"
        f"Vegetation Change (NDVI): {sign_veg}{veg_diff:,.0f} m²\n"
        f"Surface Water Change (NDWI): {sign_water}{water_diff:,.0f} m²"
    )
    # Draw stats in yellow on the dark grey bottom panel (no outline needed)
    draw.text((50, 620), stats_text, font=font_small, fill="yellow")
    
    output_path = "before_after_analysis.png"
    composite.save(output_path)
    print(f"Analysis complete! Saved to {output_path}")


if __name__ == "__main__":
    # Specify the path to your image here:
    INPUT_IMAGE_PATH = "/Users/beerendrasingh/Desktop/AWS_Data_Processing/image_process_update/Vignette_Dakar_with_exif.JPG"  # Replace with your actual image path
    
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Could not find image at '{INPUT_IMAGE_PATH}'")
        sys.exit(1)
        
    generate_infographic(INPUT_IMAGE_PATH)