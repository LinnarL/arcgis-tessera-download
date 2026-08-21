# -*- coding: utf-8 -*-
"""
TesseraToGDB.pyt

Hämtar Tessera-embeddings (rasterdata) för en bounding box och skriver dem till
en filgeodatabas.

Tessera är en foundation model för jordobservation (Sentinel-1 + Sentinel-2) som
publicerar globala, årsvisa embeddings: 128 kanaler per pixel i 10 m upplösning.
Data hämtas som vanliga HTTPS-filer från Source Cooperative:

    npy/{version}/{år}/grid_{lon}_{lat}/grid_{lon}_{lat}.npy         int8  (H, W, 128)
    npy/{version}/{år}/grid_{lon}_{lat}/grid_{lon}_{lat}_scales.npy  float32 (H, W)
    landmasks/{version}/grid_{lon}_{lat}.tiff                        georeferering

Embeddings lagras kvantiserade som int8 med en skalfaktor per pixel. Det verkliga
värdet är int8 * scale, vilket verktyget räknar ut (dekvantisering) innan data
skrivs som 32-bitars float-raster.

Rutnätet är 0,1 x 0,1 grader med tile-centrum på k*0,1 + 0,05. Varje tile ligger i
sin egen UTM-zon och saknar georeferering i .npy-filen — koordinatsystem och
origo läses därför ur tilens landmask-GeoTIFF. Tiles över öppet vatten finns inte
alls (HTTP 404).

En tile är ca 90 MB nedladdat och ca 360 MB som 32-bitars float med alla 128 band.
Verktyget visar därför storleken innan något hämtas, både i dialogen och i
körningens första meddelanden.

Källa : https://data.source.coop/tessera/tessera  (https://geotessera.org/)
Krav  : ArcGIS Pro 3.x (arcpy). Inga paket utöver Pythons standardbibliotek.
"""

import concurrent.futures
import math
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

import arcpy

# ── Konstanter ────────────────────────────────────────────────────────────────

SWEREF99TM_WKID = 3006
WGS84_WKID = 4326

BASE_URL = "https://data.source.coop/tessera/tessera"

TILE_DEG = 0.1          # tile-sida i grader
CELL_SIZE_M = 10.0      # upplösning i meter
N_BANDS = 128           # kanaler per pixel
NPY_HEADER_BYTES = 128  # .npy v1.0-header för dessa filer
LANDMASK_BYTES = 15_000  # ungefärlig storlek, försumbar men räknas med

# Dataset-versioner: etikett -> (katalog i npy/, katalog i landmasks/)
DATASETS = {
    "v1 (global, 2017-2025)":      ("v1", "v1"),
    "v2 beta (delvis täckning)":   ("v2-2B-L~beta1", "v2"),
    "v1.1 Cambridge (regional)":   ("v1.1-cam", "v1.1"),
}
DEFAULT_DATASET = "v1 (global, 2017-2025)"

YEARS = [str(y) for y in range(2017, 2026)]
DEFAULT_YEAR = "2024"

MODE_MOSAIC = "Mosaik — en sammanfogad raster i valt koordinatsystem"
MODE_TILES = "En raster per tile — originalprojektion (UTM), ingen omsampling"

# Skydd mot orimligt stora uttag. Bounding boxens sida i meter.
_MAX_SANE_SIDE_M = 200_000

# Mappar som synkas till molnet — olämpliga som cache för flera GB rådata
_SYNC_HINTS = ("onedrive", "sharepoint", "dropbox", "google drive")

_CACHE_DIRNAME = "Tessera_nedladdning"
_SCRATCH_DIRNAME = "Tessera_arbetsmapp"

_HTTP_TIMEOUT = 120
_HEAD_WORKERS = 8
_USER_AGENT = "TesseraToGDB.pyt (ArcGIS Pro)"

# Timeouten gäller varje enskild läsning, inte hela överföringen, så en stor
# chunk kan hinna slå i taket på en långsam förbindelse innan den är full.
# 1 MB åt gången tål ned till ungefär 9 kB/s innan det blir timeout.
_READ_CHUNK = 1024 * 1024

# Nedladdningen görs om vid tillfälliga fel. En avbruten fil återupptas med
# Range-huvud i stället för att hämtas om från början; servern stöder det.
_DOWNLOAD_ATTEMPTS = 4
_RETRY_STATUS = (408, 429, 500, 502, 503, 504)

# Cache för storleksuppslagningar, delad mellan updateParameters och execute.
# Nyckel: (npy-katalog, år, lon, lat) -> (bytes eller None om tilen saknas)
_size_cache = {}


def _sr(wkid):
    return arcpy.SpatialReference(wkid)


def _sr_is_valid(sr):
    """
    Är sr ett användbart koordinatsystem?

    factoryCode duger inte ensamt som test: ett eget definierat koordinatsystem
    (t.ex. en lokal transversal Mercator) har koden 0 men är fullt giltigt, medan
    ett tomt SpatialReference också har koden 0. Det som skiljer dem är att det
    tomma saknar WKT-definition.
    """
    if sr is None:
        return False
    try:
        if sr.factoryCode:
            return True
        return bool(sr.exportToString())
    except Exception:
        return False


def _coerce_sr(spec):
    """
    Bygg ett arcpy.SpatialReference av det som en parameter lämnar ifrån sig:
    ett färdigt objekt, ett EPSG-nummer, ett namn eller en WKT-sträng.

    GPCoordinateSystem lämnar sitt värde som WKT2 ("PROJCRS[...]"), och den
    strängen kan SpatialReference-konstruktorn inte läsa — den kastar
    "Error in CreateFromFile". loadFromString klarar både WKT2 och äldre WKT,
    så den används som reserv. Utan det steget faller ett valt koordinatsystem
    tillbaka på standardvärdet utan att någon varnas.
    """
    if spec is None:
        return None
    if isinstance(spec, arcpy.SpatialReference):
        return spec if _sr_is_valid(spec) else None
    if isinstance(spec, int):
        try:
            sr = arcpy.SpatialReference(spec)
            return sr if _sr_is_valid(sr) else None
        except Exception:
            return None

    text = spec if isinstance(spec, str) else str(spec)
    text = text.strip()
    if not text:
        return None
    try:
        sr = arcpy.SpatialReference(text)
        if _sr_is_valid(sr):
            return sr
    except Exception:
        pass
    try:
        sr = arcpy.SpatialReference()
        sr.loadFromString(text)
        if _sr_is_valid(sr):
            return sr
    except Exception:
        pass
    return None


def _sr_label(sr):
    """Namn på ett koordinatsystem för felmeddelanden."""
    try:
        name = sr.name or "okänt"
    except Exception:
        return "okänt"
    try:
        if sr.factoryCode:
            return "{} (EPSG:{})".format(name, sr.factoryCode)
    except Exception:
        pass
    return name


def _sr_variants(sr):
    """
    Samma koordinatsystem uttryckt på de sätt projectAs accepterar, i tur och
    ordning: objektet, EPSG-koden som text och WKT-strängen.

    projectAs bygger om koordinatsystemet internt och kan misslyckas med
    "CreateObject error creating spatial reference" för en variant men fungera
    med en annan, så alla prövas innan felet rapporteras.
    """
    variants = [sr]
    try:
        if sr.factoryCode:
            variants.append(str(sr.factoryCode))
    except Exception:
        pass
    try:
        wkt = sr.exportToString()
        if wkt:
            variants.append(wkt)
    except Exception:
        pass
    return variants


def _project_geometry(geom, target_sr):
    """Projicera en geometri till target_sr, med samtliga varianter som reserv."""
    problems = []
    for variant in _sr_variants(target_sr):
        try:
            return geom.projectAs(variant)
        except Exception as exc:
            problems.append(str(exc))
    raise ValueError(
        "Kunde inte omvandla området från {} till {}. Välj ett annat "
        "koordinatsystem för bounding boxen. ({})".format(
            _sr_label(geom.spatialReference), _sr_label(target_sr),
            problems[0] if problems else "okänt fel"
        )
    )


# =============================================================================
# Rutnät och URL:er
# =============================================================================

def _tile_center(index):
    """Tile-centrum för ett heltalsindex: index 180 -> 18.05."""
    return round(index * TILE_DEG + TILE_DEG / 2.0, 2)


def _tiles_for_bbox(west, south, east, north):
    """
    Tile-centrum (lon, lat) för alla tiles som överlappar en bounding box i
    WGS84. Tile med index i täcker [i*0,1, (i+1)*0,1); en box som precis tangerar
    en tile-kant tar alltså inte med tilen på andra sidan.
    """
    i_min = int(math.floor(west * 10))
    i_max = int(math.ceil(east * 10)) - 1
    j_min = int(math.floor(south * 10))
    j_max = int(math.ceil(north * 10)) - 1

    # En nollbred box ger i_max < i_min — ta då åtminstone med den egna tilen.
    i_max = max(i_max, i_min)
    j_max = max(j_max, j_min)

    tiles = []
    for j in range(j_min, j_max + 1):
        for i in range(i_min, i_max + 1):
            tiles.append((_tile_center(i), _tile_center(j)))
    return tiles


def _grid_name(lon, lat):
    """Filnamnsstammen för en tile, t.ex. 'grid_18.05_59.35'."""
    return "grid_{:.2f}_{:.2f}".format(lon, lat)


def _embedding_url(npy_dir, year, lon, lat):
    name = _grid_name(lon, lat)
    return "{}/npy/{}/{}/{}/{}.npy".format(BASE_URL, npy_dir, year, name, name)


def _scales_url(npy_dir, year, lon, lat):
    name = _grid_name(lon, lat)
    return "{}/npy/{}/{}/{}/{}_scales.npy".format(BASE_URL, npy_dir, year, name, name)


def _landmask_url(lm_dir, lon, lat):
    return "{}/landmasks/{}/{}.tiff".format(BASE_URL, lm_dir, _grid_name(lon, lat))


def _tile_token(value):
    """Kortform av en tile-koordinat för featureklass-/rasternamn: 18.05 -> 1805."""
    return ("m" if value < 0 else "") + "{:.2f}".format(abs(value)).replace(".", "")


# =============================================================================
# HTTP
# =============================================================================

def _request(url, method="GET"):
    return urllib.request.Request(url, headers={"User-Agent": _USER_AGENT}, method=method)


def _head_size(url):
    """
    Content-Length för en URL, eller None om filen inte finns (HTTP 404).
    Andra HTTP-fel skickas vidare — de beror på nätverk eller server, inte på
    att tilen saknas, och ska inte tolkas som "ingen data här".
    """
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=_HTTP_TIMEOUT) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _remove_quietly(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _head_etag(url):
    """Serverns ETag för en URL, eller None om den inte går att hämta."""
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=_HTTP_TIMEOUT) as resp:
            return resp.headers.get("ETag")
    except Exception:
        return None


def _resume_is_safe(url, stamp_path):
    """
    Går det att bygga vidare på en halvfärdig fil?

    En .part-fil vars längd råkar vara giltig men vars innehåll hör till en
    äldre version av filen skulle annars ge en trasig raster som klarar
    storlekskontrollen. Servern stöder Range men struntar i If-Range, så
    jämförelsen får göras här: ETag:en sparas bredvid .part-filen när
    nedladdningen börjar och måste stämma innan resten hämtas.
    """
    try:
        with open(stamp_path, "r", encoding="utf-8") as fh:
            stored = fh.read().strip()
    except OSError:
        return False
    if not stored:
        return False
    current = _head_etag(url)
    return bool(current) and current.strip() == stored


def _write_stamp(stamp_path, etag):
    if not etag:
        _remove_quietly(stamp_path)
        return
    try:
        with open(stamp_path, "w", encoding="utf-8") as fh:
            fh.write(etag.strip())
    except OSError:
        pass


def _download(url, target, expected_size=None, progress=None):
    """
    Hämta url till target. Redan hämtade filer återanvänds: när storleken är känd
    krävs att den stämmer, annars räcker att filen finns och inte är tom (gäller
    landmaskerna, som är små och statiska).
    progress är en funktion som tar antal nya bytes, för förloppsindikatorn.
    """
    if os.path.isfile(target):
        actual = os.path.getsize(target)
        if (expected_size is None and actual > 0) or actual == expected_size:
            if progress:
                progress(expected_size if expected_size is not None else actual)
            return target, True

    folder = os.path.dirname(target)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    # Delvis hämtade filer ligger kvar som .part och återupptas, både mellan
    # försöken nedan och mellan körningar. En tile är närmare 90 MB, och att
    # börja om från noll varje gång en förbindelse hackar gör stora uttag
    # praktiskt taget omöjliga.
    tmp = target + ".part"
    stamp = tmp + ".etag"
    done = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    if done and expected_size is not None and done > expected_size:
        done = 0
    if done and not _resume_is_safe(url, stamp):
        # Okänd eller ändrad ETag: filen kan höra till en äldre version.
        _remove_quietly(tmp)
        _remove_quietly(stamp)
        done = 0
    if done and progress:
        progress(done)

    for attempt in range(_DOWNLOAD_ATTEMPTS):
        last = attempt == _DOWNLOAD_ATTEMPTS - 1
        try:
            request = _request(url)
            if done:
                request.add_header("Range", "bytes={}-".format(done))

            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as resp:
                resuming = getattr(resp, "status", None) == 206
                if done and not resuming:
                    # Servern struntade i Range och skickar hela filen igen.
                    if progress:
                        progress(-done)
                    done = 0
                if not resuming:
                    _write_stamp(stamp, resp.headers.get("ETag"))
                with open(tmp, "ab" if resuming else "wb") as dst:
                    while True:
                        chunk = resp.read(_READ_CHUNK)
                        if not chunk:
                            break
                        dst.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(len(chunk))

            if expected_size is not None and done != expected_size:
                # Fel storlek betyder att .part inte hör ihop med filen på
                # servern. Kasta den och börja om i stället för att bygga vidare.
                _remove_quietly(tmp)
                _remove_quietly(stamp)
                done = 0
                raise IOError(
                    "Nedladdningen av {} blev ofullständig.".format(
                        os.path.basename(target))
                )

            os.replace(tmp, target)
            _remove_quietly(stamp)
            return target, False

        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_STATUS or last:
                raise
        except (urllib.error.URLError, OSError) as exc:
            if last:
                raise ValueError(
                    "Nedladdningen av {} avbröts: {}. {} av {} hämtat, det ligger kvar "
                    "i cachen och körningen fortsätter där den slutade nästa gång.".format(
                        os.path.basename(target), exc, _human_size(done),
                        _human_size(expected_size) if expected_size else "okänd storlek")
                )
            done = os.path.getsize(tmp) if os.path.isfile(tmp) else 0

        time.sleep(2 ** attempt)

    return target, False


def _http_error_msg(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTP-fel {} från Tessera-servern: {}".format(exc.code, exc.reason)
    if isinstance(exc, urllib.error.URLError):
        return ("Kunde inte nå Tessera-servern ({}). Kontrollera internetanslutning "
                "och eventuell proxy.".format(exc.reason))
    if isinstance(exc, TimeoutError):
        return ("Servern svarade inte i tid. Halvfärdiga filer ligger kvar i cachen, "
                "så en ny körning fortsätter där den slutade.")
    return "Fel vid anrop till Tessera-servern: {}".format(exc)


# =============================================================================
# Storleksuppskattning
# =============================================================================

def _duration(seconds):
    """Sekunder som lasbar text: '42 s' eller '3 min 20 s'."""
    seconds = max(float(seconds), 0.0)
    if seconds < 90:
        return "{:.0f} s".format(seconds)
    return "{:.0f} min {:.0f} s".format(seconds // 60, seconds % 60)


def _human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return "{:.1f} {}".format(size, unit)
        size /= 1024.0


def _scales_bytes(embedding_bytes):
    """
    Storleken på tilens scales-fil, härledd ur embeddingens storlek.

    Embeddingen är (H, W, 128) int8 och scales (H, W) float32, båda med samma
    128-byte .npy-header, så H*W = (bytes - header) / 128. Ett HEAD-anrop per
    tile räcker alltså för att veta exakt hur mycket som ska hämtas.
    """
    pixels = (embedding_bytes - NPY_HEADER_BYTES) // N_BANDS
    return pixels * 4 + NPY_HEADER_BYTES


def _tile_pixels(embedding_bytes):
    return (embedding_bytes - NPY_HEADER_BYTES) // N_BANDS


def _approx_tile_bytes(lat):
    """
    Ungefärlig storlek på en embedding-tile på en given latitud, för den
    ögonblickliga uppskattningen i dialogen.

    En tile är 0,1 grader i 10 m-celler. Höjden är i praktiken ~1130 celler och
    bredden krymper med cos(lat). Faktorn 1,07 kommer från uppmätta tiles, där
    rutorna har en liten marginal utöver den nominella storleken.
    """
    height = TILE_DEG * 111_320.0 / CELL_SIZE_M * 1.02
    width = TILE_DEG * 111_320.0 * math.cos(math.radians(lat)) / CELL_SIZE_M * 1.07
    pixels = max(height * width, 1.0)
    return int(pixels * (N_BANDS + 4)) + 2 * NPY_HEADER_BYTES


def _lookup_sizes(npy_dir, year, tiles, messages=None):
    """
    Exakt nedladdningsstorlek per tile via ett HEAD-anrop per tile.
    Returnerar {(lon, lat): bytes eller None om tilen saknas}.
    """
    result = {}
    missing = []
    for lon, lat in tiles:
        key = (npy_dir, year, lon, lat)
        if key in _size_cache:
            result[(lon, lat)] = _size_cache[key]
        else:
            missing.append((lon, lat))

    if not missing:
        return result

    def probe(tile):
        lon, lat = tile
        return tile, _head_size(_embedding_url(npy_dir, year, lon, lat))

    workers = min(_HEAD_WORKERS, len(missing))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for tile, size in pool.map(probe, missing):
            _size_cache[(npy_dir, year) + tile] = size
            result[tile] = size

    return result


def _download_bytes(sizes):
    """Totalt antal byte att hämta för {(lon, lat): embedding-bytes}."""
    total = 0
    for size in sizes.values():
        if size:
            total += size + _scales_bytes(size) + LANDMASK_BYTES
    return total


def _gdb_bytes(sizes, n_bands, mode, bbox_target):
    """
    Ungefärlig storlek på resultatet i geodatabasen.

    I mosaikläge styrs storleken av bounding boxens yta i utdatans
    koordinatsystem; i tile-läge av tilarnas egna pixelantal.
    """
    if mode == MODE_MOSAIC and bbox_target is not None:
        xmin, ymin, xmax, ymax = bbox_target
        pixels = (max(xmax - xmin, 0) / CELL_SIZE_M) * (max(ymax - ymin, 0) / CELL_SIZE_M)
    else:
        pixels = sum(_tile_pixels(s) for s in sizes.values() if s)
    return int(pixels * n_bands * 4)


def _scratch_bytes(sizes, n_bands, mode):
    """
    Tillfälligt diskutrymme under körningen.

    I mosaikläge skrivs varje tile först i sin egen projektion och projiceras
    sedan om. Den oprojicerade filen tas bort direkt efter omprojiceringen, så
    toppen är alla omprojicerade tiles plus en oprojicerad.
    """
    if mode != MODE_MOSAIC:
        return 0
    per_tile = [_tile_pixels(s) * n_bands * 4 for s in sizes.values() if s]
    if not per_tile:
        return 0
    return int(sum(per_tile) + max(per_tile))


# =============================================================================
# Bounding box
# =============================================================================

def _extent_from_value(value, fallback_sr):
    """
    arcpy.Extent ur en GPExtent-parameter. Värdet kan vara ett Extent-objekt
    eller strängen "xmin ymin xmax ymax"; saknar det ett användbart
    koordinatsystem används fallback_sr.
    """
    if value is None:
        return None

    if isinstance(value, str):
        parts = value.replace(",", ".").split()
        # Pro kan lägga till koordinatsystemets namn efter de fyra talen.
        nums = []
        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                break
        if len(nums) < 4:
            return None
        corners, sr = nums[:4], None
    else:
        try:
            corners = [float(getattr(value, name))
                       for name in ("XMin", "YMin", "XMax", "YMax")]
        except (AttributeError, TypeError, ValueError):
            return None
        sr = _coerce_sr(getattr(value, "spatialReference", None))

    # Utbredningen byggs alltid om till ett riktigt arcpy.Extent. GPExtent lämnar
    # ifrån sig ett "geoprocessing extent object" som varken bär koordinatsystem
    # eller ger en arcpy-geometri via .polygon — projicering av den geometrin
    # misslyckas med "CreateObject error creating spatial reference".
    if sr is None:
        sr = _coerce_sr(fallback_sr) or _sr(SWEREF99TM_WKID)
    return arcpy.Extent(corners[0], corners[1], corners[2], corners[3],
                        spatial_reference=sr)


def _extent_polygon(ext, sr):
    """
    Rektangeln som arcpy.Polygon med sr uttryckligen påsatt.

    Extent.polygon ärver utbredningens koordinatsystem, men om det saknas blir
    resultatet en geometri utan koordinatsystem — och projectAs returnerar då
    indata oförändrat i stället för att larma. Polygonen byggs därför här med
    ett koordinatsystem som redan är kontrollerat.
    """
    array = arcpy.Array([
        arcpy.Point(ext.XMin, ext.YMin),
        arcpy.Point(ext.XMin, ext.YMax),
        arcpy.Point(ext.XMax, ext.YMax),
        arcpy.Point(ext.XMax, ext.YMin),
        arcpy.Point(ext.XMin, ext.YMin),
    ])
    return arcpy.Polygon(array, sr)


def _project_extent(ext, target_sr):
    """
    Projicera en utbredning. Rektangelns kanter förtätas först, så att den
    projicerade utbredningen omsluter hela originalrutan även när kanterna böjs.
    """
    sr = ext.spatialReference
    if not _sr_is_valid(sr):
        raise ValueError(
            "Bounding boxen saknar koordinatsystem. Ange ett under "
            "'Koordinatsystem för bounding boxen'."
        )
    if not _sr_is_valid(target_sr):
        raise ValueError(
            "Målkoordinatsystemet är ogiltigt. Välj ett koordinatsystem för mosaiken."
        )
    try:
        if sr.factoryCode and sr.factoryCode == target_sr.factoryCode:
            return ext
    except Exception:
        pass

    poly = _extent_polygon(ext, sr)
    span = max(ext.width, ext.height)
    if span > 0:
        try:
            poly = poly.densify("DISTANCE", span / 50.0)
        except Exception:
            pass
    return _project_geometry(poly, target_sr).extent


def _snap_extent(ext, cell=CELL_SIZE_M):
    """Utvidga en utbredning till närmaste hela cellstorlek."""
    return arcpy.Extent(
        math.floor(ext.XMin / cell) * cell,
        math.floor(ext.YMin / cell) * cell,
        math.ceil(ext.XMax / cell) * cell,
        math.ceil(ext.YMax / cell) * cell,
        spatial_reference=ext.spatialReference,
    )


# =============================================================================
# Band
# =============================================================================

def _parse_bands(text):
    """
    Tolka en bandangivelse som "1-16,64" till nollbaserade index.
    Tom sträng ger alla 128 band. Banden numreras 1-128 i dialogen.
    """
    text = (text or "").strip()
    if not text:
        return list(range(N_BANDS))

    if not re.fullmatch(r"[0-9,\-\s]+", text):
        raise ValueError(
            "Ogiltig bandangivelse: '{}'. Ange band som t.ex. 1-16,64.".format(text)
        )

    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            if first > last:
                raise ValueError("Ogiltigt bandintervall: '{}'.".format(part))
            values = range(first, last + 1)
        else:
            values = [int(part)]
        for value in values:
            if not 1 <= value <= N_BANDS:
                raise ValueError(
                    "Band {} finns inte — Tessera har band 1-{}.".format(value, N_BANDS)
                )
            if value - 1 not in indices:
                indices.append(value - 1)

    if not indices:
        raise ValueError("Ingen giltig bandangivelse.")
    return sorted(indices)


# =============================================================================
# Rasterbygge
# =============================================================================

def _tile_raster(emb_path, scales_path, landmask_path, band_indices, out_path):
    """
    Bygg en 32-bitars float-raster ur en tiles npy-filer och skriv den till
    out_path. Georefereringen (koordinatsystem, origo) tas ur landmasken, som är
    den enda källan till den — .npy-filerna innehåller bara pixelvärden.
    """
    landmask = arcpy.Raster(landmask_path)
    sr = landmask.spatialReference
    if sr is None or sr.factoryCode == 0:
        raise ValueError(
            "Landmasken {} saknar koordinatsystem.".format(os.path.basename(landmask_path))
        )

    quantized = np.load(emb_path, mmap_mode="r")
    scales = np.load(scales_path)

    if quantized.shape[:2] != (landmask.height, landmask.width):
        raise ValueError(
            "Tilen {} matchar inte sin landmask ({}x{} mot {}x{}).".format(
                os.path.basename(emb_path), quantized.shape[1], quantized.shape[0],
                landmask.width, landmask.height
            )
        )
    if scales.shape[:2] != quantized.shape[:2]:
        raise ValueError(
            "Skalfilen {} matchar inte embeddingen.".format(os.path.basename(scales_path))
        )

    # Dekvantisera band för band i en färdigallokerad array: en (band, rad,
    # kolumn)-array är vad NumPyArrayToRaster vill ha, och bandvis beräkning
    # håller minnesanvändningen nere jämfört med att skala hela kuben på en gång.
    if scales.ndim == 2:
        scales = scales[:, :, np.newaxis]
    height, width = quantized.shape[0], quantized.shape[1]
    out = np.empty((len(band_indices), height, width), dtype=np.float32)
    for position, band in enumerate(band_indices):
        scale = scales[:, :, 0] if scales.shape[2] == 1 else scales[:, :, band]
        out[position] = quantized[:, :, band].astype(np.float32) * scale

    previous_sr = arcpy.env.outputCoordinateSystem
    arcpy.env.outputCoordinateSystem = sr
    try:
        raster = arcpy.NumPyArrayToRaster(
            out,
            arcpy.Point(landmask.extent.XMin, landmask.extent.YMin),
            landmask.meanCellWidth,
            landmask.meanCellHeight,
        )
        raster.save(out_path)
    finally:
        arcpy.env.outputCoordinateSystem = previous_sr

    del out, quantized, scales
    return out_path


def _snap_grid_raster(scratch_dir, target_sr, origin):
    """
    En liten raster som ProjectRaster snappar mot, så att alla tiles hamnar på
    samma 10 m-rutnät. Utan den får varje tile ett eget origo och mosaiken blir
    omsamplad en gång till.
    """
    path = os.path.join(scratch_dir, "snapgrid.tif")
    previous_sr = arcpy.env.outputCoordinateSystem
    arcpy.env.outputCoordinateSystem = target_sr
    try:
        raster = arcpy.NumPyArrayToRaster(
            np.zeros((2, 2), dtype=np.uint8),
            arcpy.Point(origin[0], origin[1]),
            CELL_SIZE_M,
            CELL_SIZE_M,
        )
        raster.save(path)
    finally:
        arcpy.env.outputCoordinateSystem = previous_sr
    return path


# =============================================================================
# Projekt- och standardvärden
# =============================================================================

def _current_map():
    """(projekt, aktiv karta), eller (None, None) utanför ett öppet projekt."""
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
    except Exception:
        return None, None
    map_obj = aprx.activeMap
    if map_obj is None:
        maps = aprx.listMaps()
        map_obj = maps[0] if maps else None
    return aprx, map_obj


def _default_gdb():
    """Projektets standardgeodatabas."""
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        if aprx.defaultGeodatabase:
            return aprx.defaultGeodatabase
    except Exception:
        pass
    workspace = arcpy.env.workspace
    if workspace and str(workspace).lower().endswith(".gdb"):
        return workspace
    return None


def _default_extent():
    """Kartvyns nuvarande utbredning, om ett projekt är öppet."""
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        view = aprx.activeView
        ext = view.camera.getExtent()
        if ext is not None and ext.XMin is not None:
            return ext
    except Exception:
        pass
    return None


def _map_sr():
    """Aktiva kartans koordinatsystem, eller None utanför ett öppet projekt."""
    try:
        _aprx, map_obj = _current_map()
        if map_obj is not None:
            sr = map_obj.spatialReference
            if _sr_is_valid(sr):
                return sr
    except Exception:
        pass
    return None


def _default_extent_crs():
    """
    Standardvärde för bounding boxens koordinatsystem: kartans eget.

    En ruta som väljs i dialogen (kartans utbredning, ett lagers utbredning
    eller en ruta man ritar) uttrycks i kartans koordinatsystem, men GPExtent
    lämnar bara fyra tal vidare — spatialReference är None. Utan kartans
    koordinatsystem som utgångspunkt tolkas talen i fel system, och ett projekt
    i t.ex. SWEREF 99 18 00 hämtar då data för fel plats utan att något larmar.
    """
    return _map_sr() or _sr(SWEREF99TM_WKID)


def _default_cache_dir():
    """
    Standardmapp för nedladdade tiles: lokal temp-mapp.

    Medvetet inte i projektmappen — den ligger ofta i OneDrive, och rådata för
    ett par tiles är hundratals MB som då skulle synkas till molnet.
    """
    return os.path.join(tempfile.gettempdir(), _CACHE_DIRNAME)


# =============================================================================
# Toolbox
# =============================================================================

class Toolbox:
    def __init__(self):
        self.label = "Tessera"
        self.alias = "tessera"
        self.tools = [HamtaTesseraRaster]


class HamtaTesseraRaster:
    def __init__(self):
        self.label = "Hämta Tessera-raster till geodatabas"
        self.description = (
            "Hämtar Tessera-embeddings för en bounding box och skriver dem till en "
            "filgeodatabas. Tessera är en foundation model för jordobservation som "
            "publicerar globala årsvisa embeddings — 128 kanaler per pixel i 10 m "
            "upplösning, byggda på Sentinel-1 och Sentinel-2.\n\n"
            "Data hämtas kvantiserat som int8 med en skalfaktor per pixel och skrivs "
            "dekvantiserat som 32-bitars float. Storleken visas innan något hämtas: "
            "en tile är ca 90 MB nedladdad och ca 360 MB i geodatabasen med alla band."
        )
        self.canRunInBackground = False
        # Senaste uppskattningen, så att den inte räknas om när användaren ändrar
        # något som inte påverkar storleken (t.ex. skriver rasternamnet).
        self._estimate_memo = (None, "")
        # Senast föreslagna rasternamnet, se updateParameters.
        self._name_memo = "tessera_{}".format(DEFAULT_YEAR)

    # ── Parametrar ────────────────────────────────────────────────────────────

    def getParameterInfo(self):
        p_extent = arcpy.Parameter(
            displayName="Bounding box (område att hämta)",
            name="extent", datatype="GPExtent",
            parameterType="Required", direction="Input",
        )
        p_extent.value = _default_extent()

        p_extent_crs = arcpy.Parameter(
            displayName="Koordinatsystem för bounding boxen (används om den saknar ett)",
            name="extent_crs", datatype="GPCoordinateSystem",
            parameterType="Optional", direction="Input",
        )
        p_extent_crs.value = _default_extent_crs()

        p_year = arcpy.Parameter(
            displayName="År",
            name="year", datatype="GPString",
            parameterType="Required", direction="Input",
        )
        p_year.filter.type = "ValueList"
        p_year.filter.list = YEARS
        p_year.value = DEFAULT_YEAR

        p_dataset = arcpy.Parameter(
            displayName="Dataset-version",
            name="dataset", datatype="GPString",
            parameterType="Required", direction="Input",
        )
        p_dataset.filter.type = "ValueList"
        p_dataset.filter.list = list(DATASETS)
        p_dataset.value = DEFAULT_DATASET

        p_bands = arcpy.Parameter(
            displayName="Band att spara, t.ex. 1-16,64 (tomt = alla 128)",
            name="bands", datatype="GPString",
            parameterType="Optional", direction="Input",
        )

        p_estimate = arcpy.Parameter(
            displayName="Uppskattad storlek",
            name="estimate", datatype="GPString",
            parameterType="Optional", direction="Input",
        )
        p_estimate.enabled = False

        p_exact = arcpy.Parameter(
            displayName="Kontrollera exakt storlek mot servern (tar några sekunder)",
            name="exact_estimate", datatype="GPBoolean",
            parameterType="Optional", direction="Input",
        )
        p_exact.value = False

        p_gdb = arcpy.Parameter(
            displayName="Utdata-geodatabas",
            name="out_gdb", datatype="DEWorkspace",
            parameterType="Required", direction="Input",
        )
        p_gdb.filter.list = ["Local Database"]
        p_gdb.value = _default_gdb()

        p_name = arcpy.Parameter(
            displayName="Namn på utdata-raster",
            name="out_name", datatype="GPString",
            parameterType="Required", direction="Input",
        )
        p_name.value = "tessera_{}".format(DEFAULT_YEAR)

        p_mode = arcpy.Parameter(
            displayName="Utdataform",
            name="out_mode", datatype="GPString",
            parameterType="Required", direction="Input", category="Utdata",
        )
        p_mode.filter.type = "ValueList"
        p_mode.filter.list = [MODE_MOSAIC, MODE_TILES]
        p_mode.value = MODE_MOSAIC

        p_target_crs = arcpy.Parameter(
            displayName="Koordinatsystem för mosaiken",
            name="target_crs", datatype="GPCoordinateSystem",
            parameterType="Optional", direction="Input", category="Utdata",
        )
        p_target_crs.value = _sr(SWEREF99TM_WKID)

        p_overwrite = arcpy.Parameter(
            displayName="Skriv över befintlig raster med samma namn",
            name="overwrite", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Utdata",
        )
        p_overwrite.value = True

        p_max_gb = arcpy.Parameter(
            displayName="Avbryt om nedladdningen överstiger (GB)",
            name="max_gb", datatype="GPDouble",
            parameterType="Optional", direction="Input", category="Nedladdning",
        )
        p_max_gb.value = 5.0

        p_cache = arcpy.Parameter(
            displayName="Cache-mapp för nedladdade tiles",
            name="cache_dir", datatype="DEFolder",
            parameterType="Optional", direction="Input", category="Nedladdning",
        )
        p_cache.value = _default_cache_dir()

        p_keep = arcpy.Parameter(
            displayName="Behåll nedladdade tiles efter körningen",
            name="keep_cache", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Nedladdning",
        )
        p_keep.value = True

        p_add = arcpy.Parameter(
            displayName="Lägg till resultatet i kartan",
            name="add_to_map", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Karta",
        )
        p_add.value = True

        return [p_extent, p_extent_crs, p_year, p_dataset, p_bands, p_estimate,
                p_exact, p_gdb, p_name, p_mode, p_target_crs, p_overwrite,
                p_max_gb, p_cache, p_keep, p_add]

    def isLicensed(self):
        return True

    # ── Dialog ────────────────────────────────────────────────────────────────

    def updateParameters(self, parameters):
        (p_extent, p_extent_crs, p_year, p_dataset, p_bands, p_estimate,
         p_exact, _p_gdb, p_name, p_mode, p_target_crs, _p_overwrite,
         _p_max_gb, _p_cache, _p_keep, _p_add) = parameters

        # Namnförslaget följer valt år tills användaren skrivit ett eget namn.
        # Jämförelsen görs mot det senast föreslagna namnet i stället för mot
        # parameterns altered-flagga, som också sätts när koden själv skriver
        # värdet här.
        suggestion = "tessera_{}".format(p_year.valueAsText or DEFAULT_YEAR)
        current = (p_name.valueAsText or "").strip()
        if current in ("", self._name_memo):
            p_name.value = suggestion
        self._name_memo = suggestion

        # Målkoordinatsystemet gäller bara mosaiken.
        p_target_crs.enabled = (p_mode.valueAsText == MODE_MOSAIC)

        # Uppskattningen är ett rent utdatafält. enabled sätts även här, inte
        # bara i getParameterInfo, eftersom Pro återställer flaggan när dialogen
        # laddas om.
        p_estimate.enabled = False

        key = (
            str(p_extent.valueAsText), str(p_extent_crs.valueAsText),
            p_year.valueAsText, p_dataset.valueAsText, p_bands.valueAsText,
            p_mode.valueAsText, str(p_target_crs.valueAsText), bool(p_exact.value),
        )
        if key != self._estimate_memo[0]:
            self._estimate_memo = (key, self._estimate_text(
                p_extent, p_extent_crs, p_year, p_dataset, p_bands, p_mode, p_target_crs,
                exact=bool(p_exact.value),
            ))
        p_estimate.value = self._estimate_memo[1]

    def _estimate_text(self, p_extent, p_extent_crs, p_year, p_dataset, p_bands,
                       p_mode, p_target_crs, exact):
        """
        Text till fältet "Uppskattad storlek".

        Utan exakt kontroll räknas storleken analytiskt ur antalet tiles, vilket
        är omedelbart men förutsätter att alla tiles finns — tiles över öppet
        vatten publiceras inte. Med exakt kontroll frågas servern om varje tile.
        """
        try:
            fallback = self._extent_crs_sr(p_extent_crs)
            ext = _extent_from_value(p_extent.value, fallback)
            if ext is None:
                return "Ange en bounding box."

            bbox = _project_extent(ext, _sr(WGS84_WKID))
            tiles = _tiles_for_bbox(bbox.XMin, bbox.YMin, bbox.XMax, bbox.YMax)
            if not tiles:
                return "Bounding boxen täcker inga tiles."

            band_indices = _parse_bands(p_bands.valueAsText)
            mode = p_mode.valueAsText or MODE_MOSAIC

            bbox_target = None
            if mode == MODE_MOSAIC:
                target_sr = self._target_sr(p_target_crs)
                target_ext = _snap_extent(_project_extent(ext, target_sr))
                bbox_target = (target_ext.XMin, target_ext.YMin,
                               target_ext.XMax, target_ext.YMax)

            if exact:
                npy_dir = DATASETS[p_dataset.valueAsText or DEFAULT_DATASET][0]
                sizes = _lookup_sizes(npy_dir, p_year.valueAsText or DEFAULT_YEAR, tiles)
                available = {t: s for t, s in sizes.items() if s}
                if not available:
                    return ("Området saknar publicerad data för {} — ingen av rutans "
                            "{} tiles finns.".format(p_year.valueAsText, len(tiles)))
                prefix = "{} av {} tiles finns".format(len(available), len(tiles))
            else:
                available = {t: _approx_tile_bytes(t[1]) for t in tiles}
                prefix = "ca {} tiles (förutsätter att alla finns)".format(len(tiles))

            download = _download_bytes(available)
            in_gdb = _gdb_bytes(available, len(band_indices), mode, bbox_target)
            scratch = _scratch_bytes(available, len(band_indices), mode)

            text = "{} — nedladdning {}, i geodatabasen {}".format(
                prefix, _human_size(download), _human_size(in_gdb)
            )
            if scratch:
                text += ", tillfälligt {}".format(_human_size(scratch))
            if len(band_indices) < N_BANDS:
                text += " ({} av {} band)".format(len(band_indices), N_BANDS)
            return text

        except ValueError as exc:
            return str(exc)
        except Exception as exc:
            return "Kunde inte uppskatta storleken: {}".format(exc)

    @staticmethod
    def _crs_param(parameter, default=None):
        """
        Koordinatsystemet ur en GPCoordinateSystem-parameter, eller default om
        den är tom. Egna koordinatsystem (faktorkod 0 men med WKT-definition)
        behålls — annars skulle koordinaterna tystlåtet tolkas som något annat.
        """
        if parameter.value is not None:
            sr = _coerce_sr(parameter.valueAsText) or _coerce_sr(parameter.value)
            if sr is not None:
                return sr
        return default or _sr(SWEREF99TM_WKID)

    @classmethod
    def _extent_crs_sr(cls, p_extent_crs):
        # Tom parameter betyder kartans koordinatsystem, inte SWEREF99 TM:
        # rutan kommer från kartan och ska tolkas i kartans system.
        return cls._crs_param(p_extent_crs, _default_extent_crs())

    @classmethod
    def _target_sr(cls, p_target_crs):
        return cls._crs_param(p_target_crs)

    def updateMessages(self, parameters):
        (p_extent, p_extent_crs, _p_year, _p_dataset, p_bands, _p_estimate,
         _p_exact, p_gdb, p_name, p_mode, p_target_crs, _p_overwrite,
         p_max_gb, p_cache, _p_keep, _p_add) = parameters

        try:
            _parse_bands(p_bands.valueAsText)
        except ValueError as exc:
            p_bands.setErrorMessage(str(exc))

        gdb = p_gdb.valueAsText
        if gdb and not gdb.lower().rstrip("\\/").endswith(".gdb"):
            p_gdb.setErrorMessage("Utdata måste vara en filgeodatabas (.gdb).")

        name = (p_name.valueAsText or "").strip()
        if name and not re.fullmatch(r"[A-Za-zÅÄÖåäö_][A-Za-zÅÄÖåäö0-9_]*", name):
            p_name.setErrorMessage(
                "Rasternamnet får bara innehålla bokstäver, siffror och understreck, "
                "och måste börja med en bokstav eller ett understreck."
            )

        # Rutan kommer från kartan men bär inget koordinatsystem med sig, så en
        # avvikelse här betyder nästan alltid att data hämtas för fel plats.
        extent_sr = self._extent_crs_sr(p_extent_crs)
        map_sr = _map_sr()
        if map_sr is not None and _sr_label(map_sr) != _sr_label(extent_sr):
            p_extent_crs.setWarningMessage(
                "Kartan använder {}. Rutan tolkas som {} — kontrollera att det är rätt, "
                "annars hämtas data för fel plats.".format(
                    _sr_label(map_sr), _sr_label(extent_sr))
            )

        ext = _extent_from_value(p_extent.value, extent_sr)
        if ext is not None:
            if ext.XMax <= ext.XMin or ext.YMax <= ext.YMin:
                p_extent.setErrorMessage("Bounding boxen har ingen yta.")
            else:
                try:
                    metric = _project_extent(
                        ext,
                        self._target_sr(p_target_crs) if p_mode.valueAsText == MODE_MOSAIC
                        else _sr(SWEREF99TM_WKID),
                    )
                    if max(metric.width, metric.height) > _MAX_SANE_SIDE_M:
                        p_extent.setWarningMessage(
                            "Rutan är över {:.0f} km på en sida. Tessera är ca 90 MB per "
                            "tile om 11 km — kontrollera storleksuppskattningen innan du "
                            "kör.".format(_MAX_SANE_SIDE_M / 1000.0)
                        )
                except Exception:
                    pass

        if p_max_gb.value is not None and p_max_gb.value <= 0:
            p_max_gb.setErrorMessage("Gränsen måste vara större än noll.")

        cache = p_cache.valueAsText
        if cache and any(hint in cache.lower() for hint in _SYNC_HINTS):
            p_cache.setWarningMessage(
                "Mappen ser ut att synkas till molnet. Nedladdade tiles är hundratals "
                "MB — välj hellre en lokal mapp, t.ex. {}.".format(_default_cache_dir())
            )

    # ── Körning ───────────────────────────────────────────────────────────────

    def execute(self, parameters, messages):
        extent_value = parameters[0].value
        fallback_sr = self._extent_crs_sr(parameters[1])
        year = parameters[2].valueAsText or DEFAULT_YEAR
        dataset = parameters[3].valueAsText or DEFAULT_DATASET
        bands_text = parameters[4].valueAsText
        out_gdb = parameters[7].valueAsText
        out_name = (parameters[8].valueAsText or "").strip()
        mode = parameters[9].valueAsText or MODE_MOSAIC
        target_sr = self._target_sr(parameters[10])
        overwrite = bool(parameters[11].value)
        max_gb = float(parameters[12].value) if parameters[12].value else 0.0
        cache_dir = parameters[13].valueAsText or _default_cache_dir()
        keep_cache = bool(parameters[14].value)
        add_to_map = bool(parameters[15].value)

        try:
            _run(extent_value, fallback_sr, year, dataset, bands_text, out_gdb,
                 out_name, mode, target_sr, overwrite, max_gb, cache_dir,
                 keep_cache, add_to_map, messages)
        except ValueError as exc:
            messages.addErrorMessage(str(exc))
            raise arcpy.ExecuteError
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            messages.addErrorMessage(_http_error_msg(exc))
            raise arcpy.ExecuteError

    def postExecute(self, parameters):
        return


# =============================================================================
# Körningens innehåll (separat funktion — går att testa utanför Pro)
# =============================================================================

def _run(extent_value, fallback_sr, year, dataset, bands_text, out_gdb, out_name,
         mode, target_sr, overwrite, max_gb, cache_dir, keep_cache, add_to_map,
         messages):
    """Utför hela hämtningen. Returnerar listan med skapade rasterdataset."""

    if dataset not in DATASETS:
        raise ValueError("Okänd dataset-version: {}".format(dataset))
    npy_dir, lm_dir = DATASETS[dataset]

    if not out_name:
        raise ValueError("Ange ett namn på utdata-rastern.")
    band_indices = _parse_bands(bands_text)

    if not arcpy.Exists(out_gdb):
        raise ValueError("Geodatabasen {} finns inte.".format(out_gdb))

    # Namnkrocken kontrolleras före nedladdningen — annars hämtas hundratals MB
    # i onödan bara för att avvisas när rastern ska skrivas.
    if mode == MODE_MOSAIC and not overwrite:
        existing = os.path.join(out_gdb, arcpy.ValidateTableName(out_name, out_gdb))
        if arcpy.Exists(existing):
            raise ValueError(
                "{} finns redan i geodatabasen. Kryssa i 'Skriv över befintlig raster' "
                "eller välj ett annat namn.".format(os.path.basename(existing))
            )

    # 1. Bounding box och tiles
    ext = _extent_from_value(extent_value, fallback_sr)
    if ext is None:
        raise ValueError("Kunde inte tolka bounding boxen.")
    if ext.XMax <= ext.XMin or ext.YMax <= ext.YMin:
        raise ValueError("Bounding boxen har ingen yta.")

    # Vilket koordinatsystem rutan tolkas i skrivs ut. GPExtent lämnar bara fyra
    # tal vidare, så tolkningen är ett antagande: syns den i loggen går det att
    # upptäcka att data hämtats för fel plats i stället för att undra efteråt.
    messages.addMessage(
        "Bounding box tolkas som {}: {:.2f}, {:.2f} till {:.2f}, {:.2f}".format(
            _sr_label(ext.spatialReference), ext.XMin, ext.YMin, ext.XMax, ext.YMax)
    )
    map_sr = _map_sr()
    if map_sr is not None and _sr_label(map_sr) != _sr_label(ext.spatialReference):
        messages.addWarningMessage(
            "Kartan använder {} men rutan tolkas som {}. Stämmer det inte hämtas data "
            "för fel plats — ändra 'Koordinatsystem för bounding boxen'.".format(
                _sr_label(map_sr), _sr_label(ext.spatialReference))
        )

    bbox = _project_extent(ext, _sr(WGS84_WKID))
    tiles = _tiles_for_bbox(bbox.XMin, bbox.YMin, bbox.XMax, bbox.YMax)
    messages.addMessage(
        "Bounding box (WGS84): {:.4f}, {:.4f} — {:.4f}, {:.4f}".format(
            bbox.XMin, bbox.YMin, bbox.XMax, bbox.YMax)
    )
    messages.addMessage(
        "{} tiles i rutan ({} band per pixel, {}).".format(
            len(tiles), len(band_indices), dataset)
    )

    # 2. Storlek — alltid innan något hämtas
    messages.addMessage("Kontrollerar storlek mot servern...")
    sizes = _lookup_sizes(npy_dir, year, tiles)
    available = {tile: size for tile, size in sizes.items() if size}
    if not available:
        raise ValueError(
            "Området saknar publicerad data för {} i {} — ingen av rutans {} tiles "
            "finns. Tiles över öppet vatten publiceras inte, och alla år finns inte "
            "i alla dataset-versioner.".format(year, dataset, len(tiles))
        )
    if len(available) < len(tiles):
        messages.addWarningMessage(
            "{} av {} tiles saknas för {} och hoppas över (öppet vatten eller ingen "
            "täckning).".format(len(tiles) - len(available), len(tiles), year)
        )

    target_ext = _snap_extent(_project_extent(ext, target_sr)) if mode == MODE_MOSAIC else None
    bbox_target = None
    if target_ext is not None:
        bbox_target = (target_ext.XMin, target_ext.YMin, target_ext.XMax, target_ext.YMax)

    download_total = _download_bytes(available)
    gdb_total = _gdb_bytes(available, len(band_indices), mode, bbox_target)
    scratch_total = _scratch_bytes(available, len(band_indices), mode)

    messages.addMessage("Att hämta      : {} ({} tiles)".format(
        _human_size(download_total), len(available)))
    messages.addMessage("I geodatabasen : ca {}".format(_human_size(gdb_total)))
    if scratch_total:
        messages.addMessage("Tillfälligt    : ca {} i arbetsmappen".format(
            _human_size(scratch_total)))

    if max_gb and download_total > max_gb * 1024 ** 3:
        raise ValueError(
            "Nedladdningen är {} och överstiger gränsen på {:.1f} GB. Minska bounding "
            "boxen eller höj gränsen under 'Nedladdning'.".format(
                _human_size(download_total), max_gb)
        )

    _check_disk_space(cache_dir, download_total, gdb_total, scratch_total, out_gdb,
                      mode, messages)

    # 3. Hämta rådata
    started = time.time()
    cached_bytes = _fetch_tiles(available, npy_dir, lm_dir, year, cache_dir, messages)
    download_secs = time.time() - started
    if cached_bytes:
        messages.addMessage(
            "Nedladdning klar: {} på {} ({}/s), resten fanns i cachen.".format(
                _human_size(cached_bytes), _duration(download_secs),
                _human_size(cached_bytes / max(download_secs, 0.001)))
        )
    else:
        messages.addMessage("Nedladdning klar: allt fanns redan i cachen.")

    # 4. Bygg raster
    started = time.time()
    if mode == MODE_MOSAIC:
        outputs = _build_mosaic(available, npy_dir, lm_dir, year, band_indices,
                                cache_dir, out_gdb, out_name, target_sr, target_ext,
                                overwrite, messages)
    else:
        outputs = _build_tiles(available, npy_dir, lm_dir, year, band_indices,
                               cache_dir, out_gdb, out_name, overwrite, messages)
    build_secs = time.time() - started

    # Att skriva och projicera om 32-bitars float tar normalt längre tid än
    # nedladdningen, särskilt med alla 128 band. Tiderna skrivs ut så att en
    # långsam körning går att placera i rätt steg i stället för att gissa.
    messages.addMessage("Tidsåtgång: nedladdning {}, rasterbygge {}.".format(
        _duration(download_secs), _duration(build_secs)))

    if not outputs:
        messages.addWarningMessage("Ingen raster skapades.")
        return outputs

    for path in outputs:
        messages.addMessage("Skrev {}".format(path))

    # 5. Karta
    if add_to_map:
        _add_to_map(outputs, messages)

    # 6. Cache
    if not keep_cache:
        messages.addMessage("Tar bort nedladdade tiles...")
        _clear_cache(available, npy_dir, lm_dir, year, cache_dir, messages)

    messages.addMessage("Klar!")
    return outputs


def _check_disk_space(cache_dir, download_total, gdb_total, scratch_total, out_gdb,
                      mode, messages):
    """Varna i förväg om någon av de berörda diskarna är för full."""
    scratch_root = os.path.dirname(cache_dir) or cache_dir
    needs = [
        (cache_dir, download_total, "cache-mappen"),
        (out_gdb, gdb_total, "geodatabasen"),
    ]
    if scratch_total:
        needs.append((scratch_root, scratch_total, "arbetsmappen"))

    for path, need, label in needs:
        probe = path
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if not probe or not os.path.isdir(probe):
            continue
        try:
            free = shutil.disk_usage(probe).free
        except OSError:
            continue
        if free < need:
            messages.addWarningMessage(
                "Bara {} ledigt på disken för {} ({}) — {} behövs.".format(
                    _human_size(free), label, probe, _human_size(need))
            )


def _tile_paths(cache_dir, npy_dir, year, lon, lat):
    """(embedding, scales, landmask) i cachen för en tile."""
    name = _grid_name(lon, lat)
    folder = os.path.join(cache_dir, npy_dir, str(year), name)
    return (
        os.path.join(folder, name + ".npy"),
        os.path.join(folder, name + "_scales.npy"),
        os.path.join(cache_dir, "landmasks", name + ".tiff"),
    )


def _fetch_tiles(available, npy_dir, lm_dir, year, cache_dir, messages):
    """
    Hämta embedding, scales och landmask för varje tile till cachen.
    Returnerar antalet byte som faktiskt passerade nätverket.
    """
    total = _download_bytes(available)
    state = {"done": 0, "network": 0, "percent": -1}

    arcpy.SetProgressor("step", "Hämtar Tessera-tiles...", 0, 100, 1)

    def progress(chunk):
        # Förloppsindikatorn uppdateras bara när heltalsprocenten ändras.
        # SetProgressorPosition går via Pros gränssnitt och är dyr nog att
        # märkas om den anropas för varje läst chunk.
        state["done"] += chunk
        if not total:
            return
        percent = min(int(state["done"] * 100 / total), 100)
        if percent != state["percent"]:
            state["percent"] = percent
            arcpy.SetProgressorPosition(percent)

    try:
        for index, (tile, emb_size) in enumerate(sorted(available.items()), start=1):
            lon, lat = tile
            emb_path, scales_path, lm_path = _tile_paths(cache_dir, npy_dir, year, lon, lat)
            arcpy.SetProgressorLabel(
                "Hämtar {} ({}/{})".format(_grid_name(lon, lat), index, len(available))
            )

            for url, path, size in (
                (_embedding_url(npy_dir, year, lon, lat), emb_path, emb_size),
                (_scales_url(npy_dir, year, lon, lat), scales_path, _scales_bytes(emb_size)),
                (_landmask_url(lm_dir, lon, lat), lm_path, None),
            ):
                try:
                    _, reused = _download(url, path, size, progress)
                except (urllib.error.URLError, OSError) as exc:
                    raise ValueError(
                        "Kunde inte hämta {}: {}".format(os.path.basename(path),
                                                         _http_error_msg(exc))
                    )
                if not reused:
                    state["network"] += os.path.getsize(path)
    finally:
        arcpy.ResetProgressor()

    return state["network"]


def _build_tiles(available, npy_dir, lm_dir, year, band_indices, cache_dir, out_gdb,
                 out_name, overwrite, messages):
    """En raster per tile i tilens egen UTM-projektion, utan omsampling."""
    outputs = []
    arcpy.SetProgressor("step", "Skriver rasterdata...", 0, len(available), 1)
    try:
        for index, tile in enumerate(sorted(available)):
            lon, lat = tile
            arcpy.SetProgressorPosition(index)
            arcpy.SetProgressorLabel("Skriver {} ({}/{})".format(
                _grid_name(lon, lat), index + 1, len(available)))

            name = arcpy.ValidateTableName(
                "{}_{}_{}".format(out_name, _tile_token(lon), _tile_token(lat)), out_gdb
            )
            out_path = os.path.join(out_gdb, name)
            if arcpy.Exists(out_path):
                if not overwrite:
                    messages.addWarningMessage(
                        "  {} finns redan — hoppas över.".format(name))
                    continue
                arcpy.management.Delete(out_path)

            emb, scales, landmask = _tile_paths(cache_dir, npy_dir, year, lon, lat)
            _tile_raster(emb, scales, landmask, band_indices, out_path)
            outputs.append(out_path)
        arcpy.SetProgressorPosition(len(available))
    finally:
        arcpy.ResetProgressor()
    return outputs


def _build_mosaic(available, npy_dir, lm_dir, year, band_indices, cache_dir, out_gdb,
                  out_name, target_sr, target_ext, overwrite, messages):
    """
    En sammanfogad raster i target_sr, klippt till bounding boxen.

    Varje tile skrivs först i sin egen UTM-projektion och projiceras sedan om mot
    ett gemensamt 10 m-rutnät (snapRaster). Utan snappningen får varje tile ett
    eget origo, och MosaicToNewRaster tvingas omsampla en gång till.
    """
    name = arcpy.ValidateTableName(out_name, out_gdb)
    out_path = os.path.join(out_gdb, name)
    if arcpy.Exists(out_path):
        if not overwrite:
            raise ValueError(
                "{} finns redan i geodatabasen. Kryssa i 'Skriv över befintlig raster' "
                "eller välj ett annat namn.".format(name)
            )
        arcpy.management.Delete(out_path)

    scratch_dir = tempfile.mkdtemp(prefix=_SCRATCH_DIRNAME + "_")

    env_extent = arcpy.env.extent
    env_snap = arcpy.env.snapRaster
    env_ocs = arcpy.env.outputCoordinateSystem
    env_cell = arcpy.env.cellSize
    env_overwrite = arcpy.env.overwriteOutput

    projected = []
    try:
        arcpy.env.overwriteOutput = True
        arcpy.env.outputCoordinateSystem = None
        arcpy.env.cellSize = None
        arcpy.env.snapRaster = _snap_grid_raster(
            scratch_dir, target_sr, (target_ext.XMin, target_ext.YMin)
        )
        arcpy.env.extent = target_ext

        arcpy.SetProgressor("step", "Bygger raster...", 0, len(available), 1)
        for index, tile in enumerate(sorted(available)):
            lon, lat = tile
            grid = _grid_name(lon, lat)
            arcpy.SetProgressorPosition(index)
            arcpy.SetProgressorLabel("Bygger {} ({}/{})".format(
                grid, index + 1, len(available)))

            emb, scales, landmask = _tile_paths(cache_dir, npy_dir, year, lon, lat)
            native = os.path.join(scratch_dir, grid + "_native.tif")
            _tile_raster(emb, scales, landmask, band_indices, native)

            reprojected = os.path.join(scratch_dir, grid + "_proj.tif")
            arcpy.management.ProjectRaster(
                native, reprojected, target_sr, "NEAREST",
                "{0} {0}".format(CELL_SIZE_M)
            )
            projected.append(reprojected)

            # Originalprojektionen behövs inte längre; att ta bort den direkt
            # halverar toppen i tillfälligt diskutrymme.
            try:
                arcpy.management.Delete(native)
            except Exception:
                pass

        arcpy.SetProgressorPosition(len(available))
        arcpy.ResetProgressor()

        if not projected:
            return []

        messages.addMessage("Sammanfogar {} tiles till {}...".format(len(projected), name))
        # FIRST i stället för BLEND: i överlappen mellan tiles ska ett helt
        # embedding-värde behållas, inte medelvärdet av två.
        arcpy.management.MosaicToNewRaster(
            projected, out_gdb, name, target_sr, "32_BIT_FLOAT",
            CELL_SIZE_M, len(band_indices), "FIRST", "FIRST"
        )
    finally:
        arcpy.ResetProgressor()
        arcpy.env.extent = env_extent
        arcpy.env.snapRaster = env_snap
        arcpy.env.outputCoordinateSystem = env_ocs
        arcpy.env.cellSize = env_cell
        arcpy.env.overwriteOutput = env_overwrite
        shutil.rmtree(scratch_dir, ignore_errors=True)

    raster = arcpy.Raster(out_path)
    messages.addMessage(
        "Mosaik: {} x {} celler, {} band, {}.".format(
            raster.width, raster.height, raster.bandCount, target_sr.name)
    )
    return [out_path]


def _add_to_map(outputs, messages):
    _aprx, map_obj = _current_map()
    if map_obj is None:
        messages.addWarningMessage("Ingen aktiv karta — resultatet lades inte till.")
        return
    for path in outputs:
        try:
            map_obj.addDataFromPath(path)
        except Exception as exc:
            messages.addWarningMessage("  Kunde inte lägga till {}: {}".format(path, exc))


def _clear_cache(available, npy_dir, lm_dir, year, cache_dir, messages):
    for tile in available:
        lon, lat = tile
        emb, scales, landmask = _tile_paths(cache_dir, npy_dir, year, lon, lat)
        for base in (emb, scales, landmask):
            for path in (base, base + ".part", base + ".part.etag"):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError as exc:
                    messages.addWarningMessage(
                        "  Kunde inte ta bort {}: {}".format(path, exc))
        folder = os.path.dirname(emb)
        try:
            if os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
        except OSError:
            pass
