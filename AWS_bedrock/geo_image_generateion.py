"""
geo_compare_local.py — v5 (sequential full-screen animation)
=============================================================
Reproduces the demo.gif animation style exactly:

  1. Wide satellite view zooms in to the location
  2. Cyan AOI box is drawn around the analysis area
  3. Each index shown ONE AT A TIME, full screen:
       TCI (real colors) → NDVI (vegetation) → NDWI (water) → NBR (damage)
  4. Each index: Before frame (hold) → crossfade → After frame (hold)
  5. Index name + legend bar clearly visible per panel
  6. Small "Before"/"After" pill bottom-left corner
  7. Stats footer per index at the bottom
  8. All in ONE single GIF file

Usage:
  python geo_compare_local.py --location "Black River, Jamaica" \\
      --before 2025-08-16 --after 2025-11-19 -o hurricane

  python geo_compare_local.py --location "Paradise, California" \\
      --before 2018-10-01 --after 2018-12-01 -o camp_fire
"""
import argparse, json, logging, math, os, tempfile, warnings
from datetime import datetime, timedelta
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from shapely.geometry import box, shape

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("geo_compare")

STAC_URL   = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
ESRI_URL   = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_SIZE  = 256

BAND_MAP = {
    "red":"red","green":"green","blue":"blue",
    "nir":"nir","nir08":"nir08","swir2":"swir22",
}

INDEX_CFG = {
    "TCI": {
        "title": "True Color (TCI)",
        "subtitle": "Real satellite imagery - reference view",
        "cmap": None, "vmin": None, "vmax": None, "mask_below": None,
        "legend_labels": ["Real colors - no index applied", "", ""],
        "legend_colors": ["#888", "", ""],
    },
    "NDVI": {
        "b1":"red","b2":"nir",
        "title": "Vegetation Index (NDVI)",
        "subtitle": "Red = deforested/bare  *  Green = healthy vegetation",
        "cmap": "RdYlGn", "vmin": -0.1, "vmax": 0.8, "mask_below": -0.1,
        "legend_labels": ["No Vegetation", "Sparse", "Dense Forest"],
        "legend_colors": ["#a50026", "#ffffbf", "#1a7416"],
    },
    "NDWI": {
        "b1":"green","b2":"nir",
        "title": "Water Index (NDWI)",
        "subtitle": "Dark blue = open water  *  Light = moist land  *  Clear = dry land",
        "cmap": "Blues", "vmin": -0.1, "vmax": 0.8, "mask_below": None,
        "legend_labels": ["Dry Land", "Moist", "Open Water"],
        "legend_colors": ["transparent", "#9ecae1", "#084594"],
    },
    "NBR": {
        "b1":"nir08","b2":"swir2",
        "title": "Damage Index (NBR)",
        "subtitle": "Red = burned/damaged  *  Yellow = stressed  *  Green = healthy",
        "cmap": "_nbr_custom", "vmin": -0.5, "vmax": 0.5, "mask_below": None,
        "legend_labels": ["Burned / Damaged", "Stressed", "Healthy"],
        "legend_colors": ["#b2182b", "#ffffbf", "#1a7416"],
    },
}

def _make_nbr_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("nbr_damage", [
        (0.0,  "#b2182b"),
        (0.35, "#ef6548"),
        (0.5,  "#ffffbf"),
        (0.65, "#74c476"),
        (1.0,  "#1a7416"),
    ])


# ─── 1. GEOCODING & METADATA EXTRACTION ─────────────────────────────────────
def extract_metadata_from_image(image_path):
    """
    Extract spatial AOI GeoDataFrame and date metadata from an image file
    (GeoTIFF, EXIF geotagged photo JPEG/PNG, or GeoJSON).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image file not found: {image_path!r}")

    ext = os.path.splitext(image_path)[1].lower()
    aoi_gdf = None
    extracted_date = None
    location_name = os.path.basename(image_path)

    # 1. GeoTIFF / TIFF
    if ext in (".tif", ".tiff"):
        log.info(f"[image] Extracting GeoTIFF spatial metadata from '{image_path}' ...")
        with rasterio.open(image_path) as src:
            if src.crs is None:
                raise ValueError(f"GeoTIFF '{image_path}' has no spatial reference system (CRS).")
            from rasterio.warp import transform_bounds
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            minx, miny, maxx, maxy = b
            aoi_gdf = gpd.GeoDataFrame([{"geometry": box(minx, miny, maxx, maxy)}], crs="EPSG:4326")
            log.info(f"[image] Extracted WGS84 bbox: ({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f})")

            tags = src.tags()
            for k in ("DATETIME", "TIFFTAG_DATETIME", "datetime", "date", "acquisition_date"):
                if k in tags and tags[k]:
                    raw_dt = str(tags[k]).replace(":", "-")[:10]
                    try:
                        datetime.strptime(raw_dt, "%Y-%m-%d")
                        extracted_date = raw_dt
                        break
                    except ValueError: pass

    # 2. Geotagged Photo (JPEG / PNG)
    elif ext in (".jpg", ".jpeg", ".png"):
        log.info(f"[image] Extracting EXIF metadata from '{image_path}' ...")
        from PIL import Image, ExifTags
        img = Image.open(image_path)
        exif = img._getexif() if hasattr(img, "_getexif") else None

        if exif:
            exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            gps_info = exif_data.get("GPSInfo")
            if gps_info:
                def _num(x):
                    if isinstance(x, tuple) and len(x) == 2:
                        return float(x[0]) / float(x[1]) if x[1] != 0 else 0.0
                    try: return float(x)
                    except: return 0.0

                def _dms_to_dd(dms, ref):
                    d = _num(dms[0]); m = _num(dms[1]); s = _num(dms[2])
                    dd = d + (m / 60.0) + (s / 3600.0)
                    return -dd if ref in ("S", "W") else dd

                lat_ref = gps_info.get(1) or gps_info.get("GPSLatitudeRef", "N")
                lat_dms = gps_info.get(2) or gps_info.get("GPSLatitude")
                lon_ref = gps_info.get(3) or gps_info.get("GPSLongitudeRef", "E")
                lon_dms = gps_info.get(4) or gps_info.get("GPSLongitude")

                if lat_dms and lon_dms:
                    lat = _dms_to_dd(lat_dms, lat_ref)
                    lon = _dms_to_dd(lon_dms, lon_ref)
                    buf = 0.005  # ~500m buffer
                    aoi_gdf = gpd.GeoDataFrame([{"geometry": box(lon - buf, lat - buf, lon + buf, lat + buf)}], crs="EPSG:4326")
                    log.info(f"[image] Extracted EXIF GPS coordinates: ({lat:.4f}, {lon:.4f})")

            for dt_key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                if dt_key in exif_data:
                    raw_dt = str(exif_data[dt_key]).replace(":", "-")[:10]
                    try:
                        datetime.strptime(raw_dt, "%Y-%m-%d")
                        extracted_date = raw_dt
                        break
                    except ValueError: pass

    # 3. JSON / GeoJSON
    elif ext in (".json", ".geojson"):
        log.info(f"[image] Reading spatial metadata from GeoJSON '{image_path}' ...")
        with open(image_path) as f:
            data = json.load(f)
        if "features" in data or data.get("type") in ("Feature", "FeatureCollection", "Polygon"):
            aoi_gdf = gpd.read_file(image_path).to_crs("EPSG:4326")
        elif "bbox" in data:
            b = data["bbox"]
            aoi_gdf = gpd.GeoDataFrame([{"geometry": box(*b)}], crs="EPSG:4326")

        dt_val = data.get("datetime") or data.get("properties", {}).get("datetime")
        if dt_val:
            extracted_date = str(dt_val)[:10]

    # Regex date fallback from filename
    if not extracted_date:
        import re
        m = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2})", os.path.basename(image_path))
        if m:
            extracted_date = m.group(1).replace("_", "-")

    # Load photo preview image if available
    photo_img = None
    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        try:
            from PIL import Image
            photo_img = Image.open(image_path).convert("RGB")
        except Exception: pass

    gps_point = None
    if 'lat' in locals() and 'lon' in locals():
        gps_point = (lat, lon)
    elif aoi_gdf is not None and not aoi_gdf.empty:
        minx, miny, maxx, maxy = aoi_gdf.total_bounds
        gps_point = ((miny + maxy) / 2.0, (minx + maxx) / 2.0)

    if aoi_gdf is None or aoi_gdf.empty:
        raise ValueError(f"Could not extract geocoordinates from '{image_path}'. "
                         f"Ensure it is a valid GeoTIFF with CRS or a geotagged photo with EXIF GPS.")

    return {
        "aoi_gdf": aoi_gdf,
        "date": extracted_date,
        "location_name": f"Image ({location_name})",
        "source_type": ext.upper()[1:],
        "photo_img": photo_img,
        "gps_point": gps_point
    }


def geocode_location(location):
    import osmnx as ox
    from geopy.geocoders import Nominatim
    log.info(f"[geocode] '{location}' ...")
    try:
        ox.settings.cache_folder = os.path.join(tempfile.gettempdir(), "osmnx_cache")
        ox.settings.use_cache = True
        r = ox.geocode_to_gdf(location)
        if r is not None and not r.empty:
            g = r.geometry.iloc[0]
            if g.geom_type in ("Polygon","MultiPolygon") and (
                    g.geom_type=="MultiPolygon" or len(g.exterior.coords)>10):
                log.info(f"[geocode] Got {g.geom_type} from OSM")
                return r
    except Exception as e:
        log.warning(f"[geocode] OSMnx failed ({e}), Nominatim fallback")
    geo = Nominatim(user_agent="geo-compare-local").geocode(location)
    if not geo: raise ValueError(f"Cannot geocode: {location!r}")
    lat, lon = geo.latitude, geo.longitude
    buf = 0.005
    log.info(f"[geocode] Nominatim fallback ({lat:.4f}, {lon:.4f})")
    return gpd.GeoDataFrame([{"geometry": box(lon-buf,lat-buf,lon+buf,lat+buf)}], crs="EPSG:4326")

def pad_bbox(aoi_gdf, pad_ratio):
    minx,miny,maxx,maxy = aoi_gdf.to_crs("EPSG:4326").total_bounds
    w,h = maxx-minx, maxy-miny
    px,py = max(w,0.002)*pad_ratio, max(h,0.002)*pad_ratio
    return gpd.GeoDataFrame([{"geometry":box(minx-px,miny-py,maxx+px,maxy+py)}],crs="EPSG:4326")


# ─── 2. STAC SEARCH ────────────────────────────────────────────────────────
def search_sentinel2(aoi_gdf, target_date, bands, max_cloud=30, days_back=60):
    import pystac_client
    end = datetime.strptime(target_date,"%Y-%m-%d").date()
    start = end - timedelta(days=days_back)
    log.info(f"[search] {start} → {end} (cloud≤{max_cloud}%)")
    client = pystac_client.Client.open(STAC_URL)
    aoi = aoi_gdf.to_crs("EPSG:4326")
    def _q(cl):
        return list(client.search(
            collections=[COLLECTION], query={"eo:cloud_cover":{"lt":cl}},
            intersects=aoi.geometry.iloc[0].__geo_interface__,
            datetime=f"{start}/{end}",
        ).items_as_dicts())
    items = _q(max_cloud) or _q(80)
    if not items: raise RuntimeError(f"No Sentinel-2 scenes for {target_date}")
    aoi_union = aoi.geometry.union_all()
    parsed = []
    for itm in items:
        cloud = itm.get("properties",{}).get("eo:cloud_cover",100)
        date  = itm.get("properties",{}).get("datetime","")
        try:    pts=itm["assets"]["thumbnail"]["href"].split("/"); tid=f"{pts[4]}{pts[5]}{pts[6]}"
        except: tid=itm.get("id","?")
        try:    cov=(aoi_union.intersection(shape(itm["geometry"])).area/aoi_union.area)*100
        except: cov=0.0
        assets={"tci":itm["assets"]["visual"]["href"]}
        for b in bands:
            an=BAND_MAP.get(b,b)
            if an in itm["assets"]: assets[b]=itm["assets"][an]["href"]
        parsed.append({"date":date,"cloud_pct":cloud,"tile_id":tid,"coverage_pct":cov,**assets})
    tile_ids={p["tile_id"] for p in parsed}
    if len(tile_ids)>1:
        cbt={t:sum(p["coverage_pct"] for p in parsed if p["tile_id"]==t)/
               len([p for p in parsed if p["tile_id"]==t]) for t in tile_ids}
        bt=max(cbt,key=cbt.get); parsed=[p for p in parsed if p["tile_id"]==bt]
        log.info(f"[search] Keeping tile {bt} ({cbt[bt]:.1f}% coverage)")
    parsed.sort(key=lambda p:(p.get("cloud_pct",100),-p.get("coverage_pct",0)))
    best=parsed[0]
    log.info(f"[search] Best: {best['date'][:10]} cloud={best.get('cloud_pct')}% cov={best.get('coverage_pct',0):.1f}%")
    return best


# ─── 3. DOWNLOAD + CLIP ────────────────────────────────────────────────────
def fetch_and_clip(href, clip_gdf, out_path):
    with rasterio.open(f"/vsicurl/{href}") as src:
        crs=src.crs
        geoms=[json.loads(clip_gdf.to_crs(crs).to_json())["features"][0]["geometry"]]
        arr,tfm=rio_mask(src,geoms,crop=True,filled=True); meta=src.meta.copy()
    meta.update(driver="GTiff",height=arr.shape[1],width=arr.shape[2],transform=tfm,compress="DEFLATE")
    with rasterio.open(out_path,"w",**meta) as dst: dst.write(arr)
    return {"path":out_path,"transform":tfm,"crs":crs}

def fetch_scene_bands(scene, needed_bands, clip_gdf, workdir, tag):
    jobs={"tci":scene["tci"],**{b:scene[b] for b in needed_bands if b in scene}}
    results={}
    with ThreadPoolExecutor(max_workers=min(len(jobs),8)) as ex:
        futs={ex.submit(fetch_and_clip,href,clip_gdf,os.path.join(workdir,f"{tag}_{n}.tif")):n
              for n,href in jobs.items()}
        for f in as_completed(futs): results[futs[f]]=f.result()
    return results


# ─── 4. INDEX MATH ─────────────────────────────────────────────────────────
def compute_index(b1_path, b2_path, out_path=None):
    if out_path is None: out_path = tempfile.mktemp(suffix=".tif")
    with rasterio.open(b1_path) as f1:
        a=f1.read(1).astype("float32"); nd1=f1.nodata
        meta=f1.meta.copy(); tfm=f1.transform; crs=f1.crs
    with rasterio.open(b2_path) as f2:
        b=f2.read(1).astype("float32"); nd2=f2.nodata
    nmask=np.zeros(a.shape,bool)
    if nd1 is not None: nmask|=(a==nd1)
    if nd2 is not None: nmask|=(b==nd2)
    nmask|=(a==0)&(b==0)
    with np.errstate(divide="ignore",invalid="ignore"): idx=(a-b)/(a+b)
    idx=np.where(np.isfinite(idx)&~nmask,idx,np.nan)
    meta.update(driver="GTiff",dtype="float32",count=1,compress="DEFLATE",nodata=np.nan)
    with rasterio.open(out_path,"w",**meta) as dst: dst.write(idx.astype("float32"),1)
    return {"array":idx,"transform":tfm,"crs":crs}


# ─── 5. RENDER PATCH → RGBA ────────────────────────────────────────────────
def index_to_rgba(arr, cfg):
    import matplotlib.cm as cm, matplotlib.colors as mc
    from PIL import Image
    cmap = _make_nbr_cmap() if cfg["cmap"]=="_nbr_custom" else cm.get_cmap(cfg["cmap"]).copy()
    cmap.set_bad((0,0,0,0))
    norm = mc.Normalize(vmin=cfg["vmin"],vmax=cfg["vmax"],clip=True)
    masked = np.ma.masked_invalid(arr.astype("float64"))
    if cfg.get("mask_below") is not None:
        masked = np.ma.masked_where(masked<cfg["mask_below"], masked)
    rgba = cmap(norm(masked))
    return Image.fromarray((rgba*255).astype("uint8"),"RGBA")

def tci_to_rgba(tci_path):
    from PIL import Image
    with rasterio.open(tci_path) as src: arr=src.read()
    rgb=np.transpose(arr[:3],(1,2,0)).astype("uint8")
    alpha=np.where(np.all(rgb==0,axis=-1),0,255).astype("uint8")
    return Image.fromarray(np.dstack([rgb,alpha]),"RGBA")


# ─── 6. ESRI BASEMAP ───────────────────────────────────────────────────────
def _ll_to_xy(lon,lat,z):
    lr=math.radians(lat); n=2**z
    return (lon+180)/360*n,(1-math.log(math.tan(lr)+1/math.cos(lr))/math.pi)/2*n

def _best_zoom(bounds,max_tiles=6,lo=2,hi=17):
    for z in range(hi,lo-1,-1):
        x0,y0=_ll_to_xy(bounds[0],bounds[3],z); x1,y1=_ll_to_xy(bounds[2],bounds[1],z)
        if max(math.floor(x1)-math.floor(x0)+1,math.floor(y1)-math.floor(y0)+1)<=max_tiles: return z
    return lo

def _fetch_tile(z,x,y):
    import urllib.request
    from PIL import Image
    req=urllib.request.Request(ESRI_URL.format(z=z,x=x,y=y),headers={
        "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer":"https://www.arcgis.com/",
    })
    with urllib.request.urlopen(req,timeout=20) as r: return Image.open(BytesIO(r.read())).convert("RGB")

def fetch_basemap(bounds, zoom):
    from PIL import Image
    minx,miny,maxx,maxy=bounds
    x0f,y0f=_ll_to_xy(minx,maxy,zoom); x1f,y1f=_ll_to_xy(maxx,miny,zoom)
    tx0,ty0=math.floor(x0f),math.floor(y0f); tx1,ty1=math.floor(x1f),math.floor(y1f)
    mosaic=Image.new("RGB",((tx1-tx0+1)*TILE_SIZE,(ty1-ty0+1)*TILE_SIZE),(25,25,28))
    ok=0
    for tx in range(tx0,tx1+1):
        for ty in range(ty0,ty1+1):
            try:    t=_fetch_tile(zoom,tx,ty); ok+=1
            except: t=Image.new("RGB",(TILE_SIZE,TILE_SIZE),(40,40,45))
            mosaic.paste(t,((tx-tx0)*TILE_SIZE,(ty-ty0)*TILE_SIZE))
    log.info(f"[basemap] {ok}/{(tx1-tx0+1)*(ty1-ty0+1)} tiles @ z{zoom} {mosaic.size[0]}×{mosaic.size[1]}")
    l,t=(x0f-tx0)*TILE_SIZE,(y0f-ty0)*TILE_SIZE
    r,bot=(x1f-tx0)*TILE_SIZE,(y1f-ty0)*TILE_SIZE
    crop=mosaic.crop((int(round(l)),int(round(t)),int(round(r)),int(round(bot))))
    def w2p(lon,lat):
        xf,yf=_ll_to_xy(lon,lat,zoom); return (xf-x0f)*TILE_SIZE,(yf-y0f)*TILE_SIZE
    return crop, w2p


# ─── 7. FONT ───────────────────────────────────────────────────────────────
def _load_font(size):
    from PIL import ImageFont
    for c in ("/System/Library/Fonts/Helvetica.ttc","/System/Library/Fonts/Arial.ttf",
              "/Library/Fonts/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "DejaVuSans-Bold.ttf","Arial.ttf"):
        try: return ImageFont.truetype(c,size)
        except: pass
    return ImageFont.load_default(size=min(size,40))

def _text_size(font, text):
    try: bb=font.getbbox(text); return bb[2]-bb[0],bb[3]-bb[1]
    except: return len(text)*(font.size//2),font.size


# ─── 7.5. PHOTO & GPS VISUAL ANCHORS ────────────────────────────────────────
def draw_gps_pin(map_img, w2p_func, lat, lon):
    """Draw a bright red location pin marker at (lat, lon) on the map_img."""
    from PIL import ImageDraw
    px, py = w2p_func(lon, lat)
    px, py = int(round(px)), int(round(py))
    
    if 0 <= px <= map_img.width and 0 <= py <= map_img.height:
        draw = ImageDraw.Draw(map_img)
        r = 12
        draw.ellipse([px - r, py - r * 2, px + r, py], fill=(235, 45, 45), outline=(255, 255, 255), width=2)
        draw.polygon([(px - r // 2, py - r // 2), (px + r // 2, py - r // 2), (px, py + r)], fill=(235, 45, 45))
        draw.ellipse([px - 4, py - int(r * 1.5), px + 4, py - int(r * 0.5)], fill=(255, 255, 255))


def embed_photo_card(canvas, photo_img, gps_point=None, card_w=200, card_h=150):
    """Overlay a styled card in top-right corner displaying the user's input photo."""
    if photo_img is None:
        return canvas

    from PIL import Image, ImageDraw

    canvas = canvas.convert("RGBA")
    aspect = photo_img.width / float(photo_img.height or 1)
    thumb_w = card_w - 16
    thumb_h = int(thumb_w / aspect)
    if thumb_h > card_h - 36:
        thumb_h = card_h - 36
        thumb_w = int(thumb_h * aspect)

    photo_thumb = photo_img.resize((thumb_w, thumb_h), resample=1)
    total_h = thumb_h + 34

    cx = canvas.width - card_w - 14
    cy = 14

    card = Image.new("RGBA", (card_w, total_h), (0, 0, 0, 0))
    cd = ImageDraw.Draw(card)

    cd.rounded_rectangle([0, 0, card_w, total_h], radius=8, fill=(12, 12, 16, 220), outline=(0, 255, 255, 200), width=2)

    font_sm = _load_font(11)
    cd.text((8, 5), "Input Photo", font=font_sm, fill=(0, 255, 255))

    card.paste(photo_thumb.convert("RGBA"), (8, 22))

    if gps_point:
        gps_txt = f"GPS: {gps_point[0]:.4f}, {gps_point[1]:.4f}"
        cd.text((8, 24 + thumb_h), gps_txt, font=_load_font(9), fill=(200, 200, 200))

    canvas.paste(card, (cx, cy), card)
    return canvas.convert("RGB")


# ─── 8. BUILD ONE FULL-SCREEN INDEX FRAME ─────────────────────────────────
def build_full_frame(basemap_img, w2p, scene_files, index_type,
                     patch_bounds, label, stats_text,
                     frame_w, frame_h,
                     input_photo_img=None, gps_point=None):
    """
    Full-screen frame for ONE index:
      - Esri basemap scaled to frame_w × (frame_h - header_h - footer_h)
      - Spectral patch at full opacity inside cyan box
      - GPS location pin marker at photo coordinates
      - Embedded Input Photo card in corner (if photo provided)
    """
    from PIL import Image, ImageDraw

    HEADER_H = 56    # title + subtitle bar at top
    LEGEND_H = 42    # colour legend at bottom
    FOOTER_H = 36    # stats text bar at very bottom
    map_h = frame_h - HEADER_H - LEGEND_H - FOOTER_H

    # ── Scale basemap to fill the map area ──
    bm = basemap_img.resize((frame_w, map_h), resample=Image.Resampling.LANCZOS)
    orig_w, orig_h = basemap_img.size
    sx, sy = frame_w / orig_w, map_h / orig_h

    def w2p_s(lon, lat):
        x, y = w2p(lon, lat)
        return x * sx, y * sy

    minx, miny, maxx, maxy = patch_bounds
    l,  t   = w2p_s(minx, maxy)
    r,  bot = w2p_s(maxx, miny)
    l, t, r, bot = int(round(l)), int(round(t)), int(round(r)), int(round(bot))
    pw, ph = max(r-l, 1), max(bot-t, 1)

    # ── Render patch ──
    if index_type == "TCI":
        patch_rgba = tci_to_rgba(scene_files["tci"]["path"])
    else:
        cfg = INDEX_CFG[index_type]
        idx = compute_index(scene_files[cfg["b1"]]["path"], scene_files[cfg["b2"]]["path"])
        patch_rgba = index_to_rgba(idx["array"], cfg)

    patch_resized = patch_rgba.resize((pw, ph), resample=Image.Resampling.LANCZOS)

    # ── Composite onto basemap ──
    map_img = bm.convert("RGBA").copy()
    map_img.paste(patch_resized, (l, t), patch_resized)
    map_img = map_img.convert("RGB")

    # ── Solid cyan box (MapView.tsx line-color #0FF, line-width 3) ──
    draw = ImageDraw.Draw(map_img)
    draw.rectangle([l, t, r, bot], outline=(0, 255, 255), width=3)

    # ── Draw GPS Location Pin Marker ──
    if gps_point:
        draw_gps_pin(map_img, w2p_s, gps_point[0], gps_point[1])

    # ── Header bar (index title + subtitle) ──
    header = Image.new("RGB", (frame_w, HEADER_H), (12, 12, 15))
    hd = ImageDraw.Draw(header)
    title_font = _load_font(max(18, int(HEADER_H * 0.42)))
    sub_font   = _load_font(max(11, int(HEADER_H * 0.24)))
    cfg_here   = INDEX_CFG[index_type]
    hd.text((16, 6),  cfg_here["title"],    font=title_font, fill=(240,240,240))
    hd.text((16, 34), cfg_here["subtitle"], font=sub_font,   fill=(160,160,160))

    # ── Legend colour bar ──
    legend = _make_legend(index_type, frame_w, LEGEND_H)

    # ── Footer stats bar ──
    footer = Image.new("RGB", (frame_w, FOOTER_H), (10, 10, 12))
    fd = ImageDraw.Draw(footer)
    sf = _load_font(max(11, FOOTER_H - 16))
    sw, sh = _text_size(sf, stats_text)
    fd.text((frame_w - sw - 16, (FOOTER_H-sh)//2), stats_text, font=sf, fill=(180,180,180))

    # ── Assemble full frame ──
    canvas = Image.new("RGB", (frame_w, frame_h), (10,10,12))
    canvas.paste(header,  (0, 0))
    canvas.paste(map_img, (0, HEADER_H))
    canvas.paste(legend,  (0, HEADER_H + map_h))
    canvas.paste(footer,  (0, HEADER_H + map_h + LEGEND_H))

    # ── "Before"/"After" pill — bottom-left of the map area (if provided) ──
    if label:
        canvas = _add_pill(canvas, label,
                           margin=12,
                           bottom_y=HEADER_H + map_h - 12)

    # ── Overlay Input Photo card in top-right corner (if provided) ──
    if input_photo_img:
        canvas = embed_photo_card(canvas, input_photo_img, gps_point)

    return canvas


def _make_legend(index_type, w, h):
    from PIL import Image, ImageDraw
    import matplotlib.cm as cm
    strip = Image.new("RGB", (w, h), (16,16,18))
    draw  = ImageDraw.Draw(strip)
    cfg   = INDEX_CFG[index_type]
    labels  = cfg["legend_labels"]

    if index_type == "TCI":
        font = _load_font(max(11, h-14))
        draw.text((16, (h-_text_size(font,labels[0])[1])//2), labels[0], font=font, fill=(140,140,140))
        return strip

    cmap = _make_nbr_cmap() if cfg["cmap"]=="_nbr_custom" else cm.get_cmap(cfg["cmap"])
    bar_x, bar_y = 16, 6
    bar_w, bar_h = int(w * 0.45), h - 12

    for px in range(bar_w):
        r2,g2,b2,_ = [int(c*255) for c in cmap(px/(bar_w-1))]
        draw.line([(bar_x+px,bar_y),(bar_x+px,bar_y+bar_h)], fill=(r2,g2,b2))

    font = _load_font(max(10, h-16))
    for i,(lbl,xpos,align) in enumerate(zip(labels,
                                             [bar_x, bar_x+bar_w//2, bar_x+bar_w],
                                             ["left","center","right"])):
        if not lbl: continue
        tw,th = _text_size(font, lbl)
        tx = xpos - (tw//2 if align=="center" else tw if align=="right" else 0)
        tx = max(bar_x, min(tx, w-tw-4))
        ty = bar_y + bar_h + 2
        draw.text((tx+1,ty+1), lbl, font=font, fill=(0,0,0))
        draw.text((tx,  ty),   lbl, font=font, fill=(200,200,200))

    return strip


def _add_pill(canvas, text, margin, bottom_y):
    from PIL import Image, ImageDraw
    img  = canvas.convert("RGBA")
    fs   = max(13, int(canvas.width * 0.018))
    font = _load_font(fs)
    tw, th = _text_size(font, text)
    pad_x, pad_y = 10, 5
    pill_w, pill_h = tw+pad_x*2, th+pad_y*2
    px = margin
    py = bottom_y - pill_h

    ov = Image.new("RGBA", img.size, (0,0,0,0))
    od = ImageDraw.Draw(ov)
    od.rounded_rectangle([px,py,px+pill_w,py+pill_h], radius=pill_h//3, fill=(0,0,0,175))
    img = Image.alpha_composite(img, ov)
    ImageDraw.Draw(img).text((px+pad_x,py+pad_y), text, font=font, fill=(255,255,255,255))
    return img.convert("RGB")


# ─── 9. STATS ──────────────────────────────────────────────────────────────
def compute_stats(arr, index_type):
    v = arr[np.isfinite(arr)]
    if not len(v): return {}
    total = len(v)
    if index_type == "NDVI":
        return {"Very Dense": float(np.sum(v>0.7)/total*100),
                "Dense": float(np.sum((v>0.5)&(v<=0.7))/total*100),
                "Light": float(np.sum((v>0)&(v<=0.5))/total*100),
                "No Vegetation": float(np.sum(v<=0)/total*100)}
    elif index_type == "NDWI":
        return {"Water": float(np.sum(v>0.3)/total*100),
                "Moist": float(np.sum((v>0.1)&(v<=0.3))/total*100),
                "Non-water": float(np.sum(v<=0.1)/total*100)}
    elif index_type == "NBR":
        return {"Healthy": float(np.sum(v>0.1)/total*100),
                "Moderate": float(np.sum((v>=-0.1)&(v<=0.1))/total*100),
                "Damaged": float(np.sum(v<-0.1)/total*100)}
    return {}

def stats_footer_text(bs, as_, index_type):
    """One-line summary of the key stat change for footer bar."""
    key = {"NDVI":"Very Dense","NDWI":"Water","NBR":"Healthy"}.get(index_type)
    if not key or not bs or not as_: return ""
    bv = bs.get(key, 0); av = as_.get(key, 0)
    delta = av - bv
    sign = "-" if delta < 0 else "+"
    return f"{key}: {bv:.1f}% -> {av:.1f}% ({sign}{abs(delta):.1f}pp) | PRE: {bv:.1f}% | POST: {av:.1f}%"


# ─── 10. ZOOM EFFECT ───────────────────────────────────────────────────────
def make_zoom_frames(basemap_img, patch_bounds, w2p, frame_w, frame_h,
                     n_frames=10, direction="in"):
    """
    Generate zoom-in (or zoom-out) frames.
    Crops the basemap from a wide view down toward the patch center.
    direction='in'  → start wide, end zoomed
    direction='out' → start zoomed, end wide
    """
    from PIL import Image

    orig_w, orig_h = basemap_img.size

    # Patch center in basemap pixel coords
    minx,miny,maxx,maxy = patch_bounds
    cx = (w2p(minx,maxy)[0] + w2p(maxx,miny)[0]) / 2
    cy = (w2p(minx,maxy)[1] + w2p(maxx,miny)[1]) / 2

    # Start: full basemap. End: tight crop around patch (50% margin each side)
    pw = w2p(maxx,miny)[0] - w2p(minx,maxy)[0]
    ph = w2p(maxx,miny)[1] - w2p(minx,maxy)[1]
    pad = max(pw, ph) * 0.6

    end_l = max(0, cx - pw/2 - pad)
    end_t = max(0, cy - ph/2 - pad)
    end_r = min(orig_w, cx + pw/2 + pad)
    end_b = min(orig_h, cy + ph/2 + pad)

    start_l, start_t, start_r, start_b = 0, 0, orig_w, orig_h

    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        if direction == "out": t = 1 - t
        # Ease: smoothstep
        t = t * t * (3 - 2 * t)
        l = int(start_l + (end_l - start_l) * t)
        top = int(start_t + (end_t - start_t) * t)
        r = int(start_r + (end_r - start_r) * t)
        bot = int(start_b + (end_b - start_b) * t)
        r = max(r, l + 10); bot = max(bot, top + 10)
        crop = basemap_img.crop((l, top, r, bot)).resize((frame_w, frame_h), resample=Image.Resampling.LANCZOS)
        frames.append(crop)

    return frames


# ─── 11. WIDE OVERVIEW FRAME (no index, just box) ──────────────────────────
def make_overview_frame(basemap_img, w2p, patch_bounds, frame_w, frame_h, location, before_date, after_date,
                        input_photo_img=None, gps_point=None):
    """First/last frame: wide view with cyan box, location label, no index overlay."""
    from PIL import Image, ImageDraw

    bm = basemap_img.resize((frame_w, frame_h), resample=Image.Resampling.LANCZOS)
    orig_w, orig_h = basemap_img.size
    sx, sy = frame_w/orig_w, frame_h/orig_h

    def s(lon, lat):
        x,y = w2p(lon, lat)
        return x*sx, y*sy

    minx,miny,maxx,maxy = patch_bounds
    l,t   = s(minx, maxy)
    r,bot = s(maxx, miny)
    l,t,r,bot = int(round(l)),int(round(t)),int(round(r)),int(round(bot))

    draw = ImageDraw.Draw(bm)
    draw.rectangle([l,t,r,bot], outline=(0,255,255), width=3)

    if gps_point:
        draw_gps_pin(bm, s, gps_point[0], gps_point[1])

    # Location label at top-left
    title_font = _load_font(max(20, int(frame_w*0.022)))
    sub_font   = _load_font(max(13, int(frame_w*0.014)))
    title = location
    sub   = f"PRE: {before_date}  *  POST: {after_date}"

    # Dark pill behind text
    tw,th = _text_size(title_font, title)
    sw,sh = _text_size(sub_font, sub)
    bar_w = max(tw,sw)+32; bar_h = th+sh+20
    ov = Image.new("RGBA", bm.size, (0,0,0,0))
    od = ImageDraw.Draw(ov)
    od.rounded_rectangle([12,12,12+bar_w,12+bar_h], radius=8, fill=(0,0,0,160))
    bm2 = Image.alpha_composite(bm.convert("RGBA"), ov).convert("RGB")
    d2 = ImageDraw.Draw(bm2)
    d2.text((28, 18),      title, font=title_font, fill=(255,255,255))
    d2.text((28, 18+th+4), sub,   font=sub_font,   fill=(180,220,180))

    if input_photo_img:
        bm2 = embed_photo_card(bm2, input_photo_img, gps_point)

    return bm2


# ─── 12. SEQUENTIAL GIF ────────────────────────────────────────────────────
def make_sequential_gif(basemap_img, w2p, fb, fa, sb, sa,
                        patch_bounds, location, before_date, after_date,
                        stats_b, stats_a, out_path,
                        frame_w=1280, frame_h=720,
                        hold_ms=1400, tr_ms=600, tr_steps=8,
                        zoom_steps=10, zoom_ms=80,
                        input_photo_img=None, gps_point=None):
    """
    Build the full sequential animation:
      [Wide overview] → [Zoom in] →
      For each index (TCI, NDVI, NDWI, NBR):
        [Before] → [crossfade] → [After] →
      [Zoom out] → loop
    """
    from PIL import Image

    frames = []
    dur    = []

    def add(img, ms): frames.append(img.convert("RGB")); dur.append(ms)

    INDICES = ["TCI","NDVI","NDWI","NBR"]

    # 1. Wide overview (hold 2s)
    log.info("[gif] Wide overview frame ...")
    overview = make_overview_frame(basemap_img, w2p, patch_bounds,
                                   frame_w, frame_h, location, before_date, after_date,
                                   input_photo_img=input_photo_img, gps_point=gps_point)
    for _ in range(3): add(overview, 700)   # 3×700ms = 2.1s

    # 2. Zoom in frames
    log.info("[gif] Zoom in ...")
    zoom_in = make_zoom_frames(basemap_img, patch_bounds, w2p,
                               frame_w, frame_h, zoom_steps, "in")
    for f in zoom_in: add(f, zoom_ms)

    # 3. For each index: Before → crossfade → After
    map_bottom = frame_h - 56 - 42 - 36 - 12  # HEADER_H=56, LEGEND_H=42, FOOTER_H=36
    for it in INDICES:
        log.info(f"[gif] Rendering {it} Before/After ...")
        bs_text = stats_footer_text(stats_b.get(it,{}), stats_a.get(it,{}), it) if it!="TCI" else \
                  f"PRE: {before_date}  |  POST: {after_date}"

        bf_base = build_full_frame(basemap_img, w2p, fb, it, patch_bounds,
                                   None, bs_text, frame_w, frame_h,
                                   input_photo_img=input_photo_img, gps_point=gps_point)
        af_base = build_full_frame(basemap_img, w2p, fa, it, patch_bounds,
                                   None, bs_text, frame_w, frame_h,
                                   input_photo_img=input_photo_img, gps_point=gps_point)

        bf = _add_pill(bf_base, "Before", margin=12, bottom_y=map_bottom)
        af = _add_pill(af_base, "After",  margin=12, bottom_y=map_bottom)

        # Hold Before
        for _ in range(2): add(bf, hold_ms // 2)
        # Crossfade Before -> After (clean pill switch without ghosting)
        step_ms = max(30, tr_ms // tr_steps)
        for i in range(1, tr_steps + 1):
            alpha = i / (tr_steps + 1)
            blended = Image.blend(bf_base.convert("RGB"), af_base.convert("RGB"), alpha)
            lbl = "Before" if alpha < 0.5 else "After"
            add(_add_pill(blended, lbl, margin=12, bottom_y=map_bottom), step_ms)
        # Hold After
        add(af, hold_ms)
        # Brief pause between indices
        add(af, 400)

    # 4. Zoom out
    log.info("[gif] Zoom out ...")
    zoom_out = make_zoom_frames(basemap_img, patch_bounds, w2p,
                                frame_w, frame_h, zoom_steps//2, "out")
    for f in zoom_out: add(f, zoom_ms*2)

    # 5. Back to overview (hold 1.5s)
    for _ in range(2): add(overview, 750)

    log.info(f"[gif] Quantizing {len(frames)} frames with per-frame adaptive palettes (no dither noise)...")
    q_frames = []
    for f in frames:
        q = f.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q_frames.append(q)

    log.info(f"[gif] Saving {len(q_frames)} crisp frames → {out_path}")
    q_frames[0].save(out_path, save_all=True, append_images=q_frames[1:],
                     duration=dur, loop=0, disposal=2)
    return out_path


# ─── 13. SUMMARY PNG ───────────────────────────────────────────────────────
def make_summary_png(basemap_img, w2p, fb, fa, sb, sa,
                     patch_bounds, location, before_date, after_date,
                     stats_b, stats_a, out_path, frame_w=1280, frame_h=720,
                     input_photo_img=None, gps_point=None):
    """4 columns side by side (one per index) as a static reference image."""
    from PIL import Image, ImageDraw

    col_w  = frame_w // 4
    INDICES = ["TCI","NDVI","NDWI","NBR"]

    cols = []
    for it in INDICES:
        f = build_full_frame(basemap_img, w2p, fb, it, patch_bounds,
                             "Before", "", col_w, frame_h,
                             input_photo_img=None, gps_point=gps_point)
        cols.append(f)

    grid = Image.new("RGB",(frame_w, frame_h),(10,10,12))
    for i,col in enumerate(cols): grid.paste(col,(i*col_w,0))

    # Title bar
    bar = Image.new("RGB",(frame_w,40),(12,12,15))
    d   = ImageDraw.Draw(bar)
    font= _load_font(16)
    t   = f"{location}  ·  PRE: {before_date}  vs  POST: {after_date}  ·  Satellite Change Analysis"
    tw,_= _text_size(font,t)
    d.text(((frame_w-tw)//2,12),t,font=font,fill=(200,200,200))
    canvas=Image.new("RGB",(frame_w,frame_h+40),(10,10,12))
    canvas.paste(bar,(0,0)); canvas.paste(grid,(0,40))

    if input_photo_img:
        canvas = embed_photo_card(canvas, input_photo_img, gps_point)

    canvas.save(out_path,quality=92)
    return out_path


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────
def run(location=None, before_date=None, after_date=None, image_path=None,
        output_base="comparison", max_cloud=30, frame_w=1280, frame_h=720,
        patch_pad=0.05, basemap_pad=0.5, tiles_across=6):

    workdir = tempfile.mkdtemp(prefix="geo_compare_")
    log.info(f"[main] workdir: {workdir}")

    photo_img = None
    gps_point = None

    if image_path:
        img_meta = extract_metadata_from_image(image_path)
        aoi_gdf  = img_meta["aoi_gdf"]
        location = location or img_meta["location_name"]
        photo_img = img_meta.get("photo_img")
        gps_point = img_meta.get("gps_point")

        if img_meta["date"]:
            log.info(f"[main] Auto-extracted acquisition date from image: {img_meta['date']}")
            if before_date and not after_date:
                after_date = img_meta["date"]
            elif after_date and not before_date:
                before_date = img_meta["date"]
            elif not before_date and not after_date:
                before_date = img_meta["date"]
                b_dt = datetime.strptime(before_date, "%Y-%m-%d")
                after_date = (b_dt + timedelta(days=90)).strftime("%Y-%m-%d")
    else:
        if not location:
            raise ValueError("Either --location or --image (-i) must be provided.")
        aoi_gdf = geocode_location(location)

    if not before_date or not after_date:
        raise ValueError("Both --before and --after dates must be specified (or inferable from --image metadata).")

    patch_gdf   = pad_bbox(aoi_gdf, pad_ratio=patch_pad)
    basemap_gdf = pad_bbox(patch_gdf, pad_ratio=basemap_pad)
    bb = tuple(basemap_gdf.total_bounds)
    pb = tuple(patch_gdf.total_bounds)

    zoom = _best_zoom(bb, max_tiles=tiles_across)
    log.info(f"[main] Fetching Esri basemap (zoom {zoom}) ...")
    bm, w2p = fetch_basemap(bb, zoom)
    log.info(f"[main] Basemap: {bm.size[0]}×{bm.size[1]} px")

    needed = ["red","green","nir","nir08","swir2"]

    log.info(f"\n[main] ── BEFORE ({before_date}) ──")
    sb = search_sentinel2(patch_gdf, before_date, needed, max_cloud=max_cloud)
    fb = fetch_scene_bands(sb, needed, patch_gdf, workdir, "before")

    log.info(f"\n[main] ── AFTER ({after_date}) ──")
    sa = search_sentinel2(patch_gdf, after_date, needed, max_cloud=max_cloud)
    fa = fetch_scene_bands(sa, needed, patch_gdf, workdir, "after")

    log.info(f"\n[main] Computing stats ...")
    stats_b, stats_a = {}, {}
    for it in ["NDVI","NDWI","NBR"]:
        cfg = INDEX_CFG[it]
        bi = compute_index(fb[cfg["b1"]]["path"],fb[cfg["b2"]]["path"])
        ai = compute_index(fa[cfg["b1"]]["path"],fa[cfg["b2"]]["path"])
        stats_b[it] = compute_stats(bi["array"],it)
        stats_a[it] = compute_stats(ai["array"],it)

    for it in ["NDVI","NDWI","NBR"]:
        print(f"\n  {INDEX_CFG[it]['title']}")
        for k in stats_b[it]:
            bv=stats_b[it][k]; av=stats_a[it][k]
            print(f"    {k}: {bv:.1f}% → {av:.1f}%  ({'▼' if av<bv else '▲'}{abs(av-bv):.1f}pp)")

    gif_path = f"{output_base}.gif"
    png_path = f"{output_base}.png"

    log.info(f"\n[main] Building sequential GIF ...")
    make_sequential_gif(bm, w2p, fb, fa, sb, sa, pb,
                        location, before_date, after_date,
                        stats_b, stats_a, gif_path,
                        frame_w=frame_w, frame_h=frame_h,
                        input_photo_img=photo_img, gps_point=gps_point)
    log.info(f"[main] GIF → {gif_path}")

    log.info(f"[main] Building summary PNG ...")
    make_summary_png(bm, w2p, fb, fa, sb, sa, pb,
                     location, before_date, after_date,
                     stats_b, stats_a, png_path,
                     frame_w=frame_w, frame_h=frame_h,
                     input_photo_img=photo_img, gps_point=gps_point)
    log.info(f"[main] PNG → {png_path}")

    import shutil; shutil.rmtree(workdir, ignore_errors=True)
    return gif_path, png_path

    if not before_date or not after_date:
        raise ValueError("Both --before and --after dates must be specified (or inferable from --image metadata).")

    patch_gdf   = pad_bbox(aoi_gdf, pad_ratio=patch_pad)
    basemap_gdf = pad_bbox(patch_gdf, pad_ratio=basemap_pad)
    bb = tuple(basemap_gdf.total_bounds)
    pb = tuple(patch_gdf.total_bounds)

    zoom = _best_zoom(bb, max_tiles=tiles_across)
    log.info(f"[main] Fetching Esri basemap (zoom {zoom}) ...")
    bm, w2p = fetch_basemap(bb, zoom)
    log.info(f"[main] Basemap: {bm.size[0]}×{bm.size[1]} px")

    needed = ["red","green","nir","nir08","swir2"]

    log.info(f"\n[main] ── BEFORE ({before_date}) ──")
    sb = search_sentinel2(patch_gdf, before_date, needed, max_cloud=max_cloud)
    fb = fetch_scene_bands(sb, needed, patch_gdf, workdir, "before")

    log.info(f"\n[main] ── AFTER ({after_date}) ──")
    sa = search_sentinel2(patch_gdf, after_date, needed, max_cloud=max_cloud)
    fa = fetch_scene_bands(sa, needed, patch_gdf, workdir, "after")

    log.info(f"\n[main] Computing stats ...")
    stats_b, stats_a = {}, {}
    for it in ["NDVI","NDWI","NBR"]:
        cfg = INDEX_CFG[it]
        bi = compute_index(fb[cfg["b1"]]["path"],fb[cfg["b2"]]["path"])
        ai = compute_index(fa[cfg["b1"]]["path"],fa[cfg["b2"]]["path"])
        stats_b[it] = compute_stats(bi["array"],it)
        stats_a[it] = compute_stats(ai["array"],it)

    for it in ["NDVI","NDWI","NBR"]:
        print(f"\n  {INDEX_CFG[it]['title']}")
        for k in stats_b[it]:
            bv=stats_b[it][k]; av=stats_a[it][k]
            print(f"    {k}: {bv:.1f}% → {av:.1f}%  ({'▼' if av<bv else '▲'}{abs(av-bv):.1f}pp)")

    gif_path = f"{output_base}.gif"
    png_path = f"{output_base}.png"

    log.info(f"\n[main] Building sequential GIF ...")
    make_sequential_gif(bm, w2p, fb, fa, sb, sa, pb,
                        location, before_date, after_date,
                        stats_b, stats_a, gif_path,
                        frame_w=frame_w, frame_h=frame_h)
    log.info(f"[main] GIF → {gif_path}")

    log.info(f"[main] Building summary PNG ...")
    make_summary_png(bm, w2p, fb, fa, sb, sa, pb,
                     location, before_date, after_date,
                     stats_b, stats_a, png_path,
                     frame_w=frame_w, frame_h=frame_h)
    log.info(f"[main] PNG → {png_path}")

    import shutil; shutil.rmtree(workdir, ignore_errors=True)
    return gif_path, png_path


def main():
    p = argparse.ArgumentParser(description="Sequential satellite change GIF — text or image input")
    p.add_argument("-i", "--image",   help="Path to input image (GeoTIFF, geotagged JPEG/PNG, or GeoJSON) with spatial metadata")
    p.add_argument("--location",     help="Location name (required if --image is not provided)")
    p.add_argument("--before",       help="YYYY-MM-DD (optional if inferable from --image)")
    p.add_argument("--after",        help="YYYY-MM-DD (optional if inferable from --image)")
    p.add_argument("--max-cloud",    type=int, default=30)
    p.add_argument("-o","--output",  default="comparison")
    p.add_argument("--width",        type=int, default=1280)
    p.add_argument("--height",       type=int, default=720)
    p.add_argument("--basemap-pad",  type=float, default=0.5)
    p.add_argument("--patch-pad",    type=float, default=0.05)
    p.add_argument("--tiles-across", type=int, default=6)
    a = p.parse_args()

    if not a.image and not a.location:
        p.error("One of --location or --image (-i) must be specified.")

    gif, png = run(location=a.location, before_date=a.before, after_date=a.after,
                   image_path=a.image, output_base=a.output,
                   max_cloud=a.max_cloud, frame_w=a.width, frame_h=a.height,
                   patch_pad=a.patch_pad, basemap_pad=a.basemap_pad, tiles_across=a.tiles_across)
    print(f"\n✅ Done:\n  GIF → {gif}\n  PNG → {png}")

if __name__ == "__main__":
    main()
