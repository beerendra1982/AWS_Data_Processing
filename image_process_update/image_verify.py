from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def extract_exif(image_path):
    img = Image.open(image_path)
    exif = img._getexif()
    if not exif:
        print("No EXIF data found.")
        return

    gps_info = {}
    for tag, value in exif.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for t in value:
                sub_tag = GPSTAGS.get(t, t)
                gps_info[sub_tag] = value[t]

    if gps_info:
        print("GPS Data Found:", gps_info)
    else:
        print("No GPS data found in EXIF.")

# Replace with your image path
extract_exif("/Users/beerendrasingh/Desktop/AWS_Data_Processing/image_process_update/Vignette_Dakar_with_exif.JPG")