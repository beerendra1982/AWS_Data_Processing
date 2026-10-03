import piexif
from PIL import Image

def add_exif_metadata(image_path, output_path, lat, lon):
    # Convert latitude and longitude to DMS format
    def to_dms(value):
        degrees = int(value)
        minutes = int((value - degrees) * 60)
        seconds = round((value - degrees - minutes / 60) * 3600, 6)
        return ((degrees, 1), (minutes, 1), (int(seconds * 100), 100))

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: to_dms(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: to_dms(abs(lon)),
    }

    exif_dict = {"GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)

    # Add EXIF data to the image
    img = Image.open(image_path)
    img.save(output_path, exif=exif_bytes)
    print(f"EXIF metadata added to {output_path}")

# Replace with your image path and desired GPS coordinates
add_exif_metadata(
    "/Users/beerendrasingh/Desktop/AWS_Data_Processing/image_process_update/Vignette_Dakar.JPG",
    "/Users/beerendrasingh/Desktop/AWS_Data_Processing/image_process_update/Vignette_Dakar_with_exif.JPG",
    lat=14.6928,  # Example latitude
    lon=-17.4467  # Example longitude
)