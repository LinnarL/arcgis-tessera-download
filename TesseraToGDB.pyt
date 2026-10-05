# -*- coding: utf-8 -*-
"""
TesseraToGDB.pyt

Hämtar Tessera-embeddings (rasterdata) för ett område och skriver dem till en
filgeodatabas. Området avgränsas antingen med polygoner (ur ett lager, där ett
urval respekteras, eller ritade i kartan) eller med en utbredning (kartvyn, ett
lagers utbredning, en ritad rektangel eller inskrivna koordinater).

Med polygoner hämtas bara de tiles som skär själva polygonerna, inte alla inom
deras bounding box, och mosaiken klipps till polygonerna (NoData utanför). I
tile-läget skrivs hela tiles utan klippning.

Tessera är en foundation model för jordobservation (Sentinel-1 + Sentinel-2) som
publicerar globala, årsvisa embeddings: 128 kanaler per pixel i 10 m upplösning.

Nedladdning och register sköts av biblioteket geotessera, som hämtar data direkt
från projektets S3-bucket. Verktyget använder geotessera för att veta vilka tiles
som finns, hur stora de är och för att hämta dem till en cache-mapp, och gör
sedan resten med arcpy och numpy: dekvantisering, omprojicering och skrivning
till geodatabasen.

Embeddings lagras kvantiserade som int8 med en skalfaktor per pixel. Det verkliga
värdet är int8 * scale, vilket verktyget räknar ut (dekvantisering) innan data
skrivs som 32-bitars float-raster.

Rutnätet är 0,1 x 0,1 grader med tile-centrum på k*0,1 + 0,05. Varje tile ligger i
sin egen UTM-zon och saknar georeferering i .npy-filen — koordinatsystem och
origo läses därför ur tilens landmask-GeoTIFF.

En tile är ca 90 MB nedladdat och ca 360 MB som 32-bitars float med alla 128 band.
Verktyget visar därför storleken innan något hämtas, både i dialogen och i
körningens första meddelanden. Storleken kommer ur geotesseras register och
kostar inga extra anrop.

Verktygstips (parameterförklaringar) skrivs till
TesseraToGDB.HamtaTesseraRaster.pyt.xml från TOOLTIPS nedan när verktygslådan
laddas, så att texten bara finns på ett ställe.

Krav : ArcGIS Pro 3.x (arcpy), numpy och geotessera.
       geotessera finns inte i standardmiljön arcgispro-py3. Klona miljön och
       installera paketet, t.ex. arcgispro-py3-personal, och peka Pro på den
       kloningen innan verktygslådan används.

Källa : https://geotessera.org/  (data via s3://tessera-embeddings)
"""

import math
import os
import re
import shutil
import tempfile
import time
import uuid
from xml.sax.saxutils import escape

import numpy as np

import arcpy

try:
    from geotessera import GeoTessera
    from geotessera import registry as gt_registry
    _GEOTESSERA_ERROR = None
except Exception as _exc:                                   # noqa: BLE001
    GeoTessera = None
    gt_registry = None
    _GEOTESSERA_ERROR = _exc

# ── Konstanter ────────────────────────────────────────────────────────────────

SWEREF99TM_WKID = 3006
WGS84_WKID = 4326

TILE_DEG = 0.1          # tile-sida i grader
CELL_SIZE_M = 10.0      # upplösning i meter
N_BANDS = 128           # kanaler per pixel
NPY_HEADER_BYTES = 128  # .npy v1.0-header för dessa filer
LANDMASK_BYTES = 15_000  # ungefärlig storlek, försumbar men räknas med

# Dataset-versioner: etikett -> (version, variant) som geotessera vill ha dem.
# v1 är den globala produktionskörningen; v1.1 Cambridge täcker bara delar av
# Europa och v2 är en beta med ojämn täckning per år.
DATASETS = {
    "v1 (global, 2017-2025)":      ("v1", "vultr"),
    "v1.1 Cambridge (regional)":   ("v1.1", "cambridge"),
    "v2 beta (delvis täckning)":   ("v2", "2B-L~beta1"),
}
DEFAULT_DATASET = "v1 (global, 2017-2025)"

YEARS = [str(y) for y in range(2017, 2026)]
DEFAULT_YEAR = "2024"

MODE_MOSAIC = "Mosaik — en sammanfogad raster i valt koordinatsystem"
MODE_TILES = "En raster per tile — originalprojektion (UTM), ingen omsampling"

# Sätt att avgränsa området. Den gamla etiketten för polygonläget accepteras
# fortfarande från skript men visas inte i listan.
AOI_POLYGONS = "Polygoner (lager eller ritade i kartan)"
AOI_EXTENT = "Utbredning (kartvy, lager eller koordinater)"
_AOI_POLYGONS_OLD = "Polygoner i ett lager"
DEFAULT_AOI_MODE = AOI_EXTENT

TOOL_SUMMARY = (
    "Hämtar Tessera-embeddings för ett område och skriver dem till en filgeodatabas. "
    "Tessera är en foundation model för jordobservation som publicerar globala årsvisa "
    "embeddings, 128 kanaler per pixel i 10 m upplösning, byggda på Sentinel-1 och "
    "Sentinel-2. Området avgränsas med polygoner (ur ett lager eller ritade i kartan) "
    "eller med en utbredning. Storleken visas innan något hämtas: en tile är ca 90 MB "
    "nedladdad och ca 360 MB i geodatabasen med alla band."
)

# Verktygstips per parameter, visas i verktygsdialogen. Se _write_tool_metadata.
TOOLTIPS = {
    "aoi_mode": (
        "Hur området avgränsas. Med polygoner hämtas bara de tiles som skär själva "
        "polygonerna, och mosaiken klipps till dem (NoData utanför). Med en utbredning "
        "hämtas allt inom en rektangel: kartvyns aktuella utbredning, ett lagers "
        "utbredning, en ritad rektangel eller inskrivna koordinater."
    ),
    "aoi": (
        "Polygoner som avgränsar området. Välj ett polygonlager i listan, eller rita "
        "polygoner direkt i kartan med pennan bredvid fältet. Har lagret ett urval används "
        "bara de valda objekten, annars alla. Polygonerna slås ihop och kan ha vilket "
        "koordinatsystem som helst. Tiles som bara ligger inom polygonernas bounding box, "
        "men inte rör själva polygonerna, hämtas inte. I mosaikläget blir allt utanför "
        "polygonerna NoData; i tile-läget skrivs de berörda tilarna hela, utan klippning."
    ),
    "extent": (
        "Rektangel som avgränsar området. I listan kan du välja kartvyns aktuella "
        "utbredning eller ett lager för att använda dess utbredning. Du kan också rita en "
        "rektangel i kartan eller skriva in koordinater. Standard är kartvyns utbredning "
        "när dialogen öppnas. Vilket koordinatsystem talen tolkades i står i meddelandena."
    ),
    "extent_crs": (
        "Koordinatsystem som utbredningens fyra tal tolkas i, när utbredningen inte bär "
        "ett eget. Inskrivna koordinater och kartvyns utbredning saknar det, ett lagers "
        "utbredning har det och då används lagrets. Standard är den aktiva kartans "
        "koordinatsystem; lämnas fältet tomt används också kartans, och utan karta "
        "SWEREF 99 TM. Fel värde ger inget fel, utan data för en annan plats. Används "
        "inte med polygoner, som har sitt eget koordinatsystem."
    ),
    "year": (
        "Året som embeddings hämtas för. Varje år är en egen årsmosaik av "
        "satellitbilderna. Alla år finns inte i alla dataset-versioner."
    ),
    "dataset": (
        "Vilken publicering av Tessera som används. v1 är den globala "
        "produktionskörningen och standard. v1.1 Cambridge täcker bara delar av Europa, "
        "v2 är en beta med ojämn täckning per år."
    ),
    "bands": (
        "Vilka av de 128 banden som sparas, numrerade 1-128. Skriv t.ex. 1-16,64. Tomt "
        "betyder alla 128. Färre band minskar storleken i geodatabasen men inte "
        "nedladdningen, eftersom hela tilen alltid hämtas."
    ),
    "estimate": (
        "Uppskattad storlek, uppdateras när du ändrar område, år, band eller utdataform. "
        "Visar nedladdning, storlek i geodatabasen och tillfälligt diskutrymme. Utan exakt "
        "kontroll förutsätts att alla tiles finns; med polygoner räknas på polygonernas "
        "bounding box, så siffran är ett tak. Går inte att redigera."
    ),
    "exact_estimate": (
        "Fråga Tessera-registret vilka tiles som faktiskt finns och hur stora de är. "
        "Tiles över öppet vatten publiceras inte, så uppskattningen blir lägre och exakt. "
        "Första gången hämtas ett manifest över alla tiles, vilket tar en stund. Av som "
        "standard."
    ),
    "out_gdb": (
        "Filgeodatabas som rastern eller rastren skrivs till. Standard är projektets "
        "standardgeodatabas. Måste vara en .gdb."
    ),
    "out_name": (
        "Namn på utdata-rastern. Standard är tessera_<år> och följer valt år tills du "
        "skriver ett eget namn. I tile-läget får varje raster tilens koordinater som "
        "tillägg, t.ex. tessera_2024_1805_5935. Bara bokstäver, siffror och understreck."
    ),
    "out_mode": (
        "Mosaik ger en sammanfogad raster i valt koordinatsystem, omsamplad till ett "
        "gemensamt 10 m-rutnät och klippt till området (till polygonerna om sådana "
        "används). En raster per tile behåller varje tiles egen UTM-projektion utan "
        "omsampling; tilarna skrivs hela och klipps inte, varken till utbredningen eller "
        "till polygonerna."
    ),
    "target_crs": (
        "Koordinatsystem för mosaiken. Standard är SWEREF 99 TM. Cellstorleken är alltid "
        "10 m. Används bara i mosaikläget."
    ),
    "overwrite": (
        "Skriv över en befintlig raster med samma namn. På som standard. Av gör att "
        "verktyget stoppar före nedladdningen om mosaiken redan finns, och hoppar över "
        "befintliga rastrar i tile-läget."
    ),
    "max_gb": (
        "Övre gräns för nedladdningen i GB. Verktyget stoppar innan något hämtas om "
        "området kräver mer. Standard 5 GB, ungefär 55 tiles."
    ),
    "cache_dir": (
        "Mapp där tiles laddas ned innan de skrivs till geodatabasen. Standard är "
        "%LOCALAPPDATA%\\Tessera_nedladdning, utanför OneDrive. Undvik mappar som synkas "
        "till molnet: en tile är ca 90 MB."
    ),
    "keep_cache": (
        "Behåll de nedladdade filerna efter körningen, så att en ny körning över samma "
        "område slipper hämta dem igen. Av som standard: filerna tas bort när rastern är "
        "skriven. Kostar ca 90 MB per tile."
    ),
    "add_to_map": (
        "Lägg till resultatet i den aktiva kartan när körningen är klar. På som standard."
    ),
}

# Skydd mot orimligt stora uttag. Bounding boxens sida i meter.
_MAX_SANE_SIDE_M = 200_000

# Mappar som synkas till molnet — olämpliga som cache för flera GB rådata
_SYNC_HINTS = ("onedrive", "sharepoint", "dropbox", "google drive")

_CACHE_DIRNAME = "Tessera_nedladdning"
_SCRATCH_DIRNAME = "Tessera_arbetsmapp"

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
        "Kunde inte omvandla området från {} till {}. Kontrollera områdets "
        "koordinatsystem. ({})".format(
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


def _tile_token(value):
    """Kortform av en tile-koordinat för featureklass-/rasternamn: 18.05 -> 1805."""
    return ("m" if value < 0 else "") + "{:.2f}".format(abs(value)).replace(".", "")


# =============================================================================
# geotessera: register och nedladdning
# =============================================================================

def _require_geotessera():
    """Ge ett begripligt fel när paketet saknas i den aktiva Python-miljön."""
    if GeoTessera is None:
        raise ValueError(
            "Paketet geotessera kunde inte laddas i den Python-miljö som ArcGIS Pro "
            "använder ({}). Klona arcgispro-py3, installera geotessera i kloningen "
            "och byt aktiv miljö i Pro under Settings, Package Manager. "
            "Ursprungligt fel: {}".format(_env_name(), _GEOTESSERA_ERROR)
        )


def _env_name():
    import sys
    return os.path.basename(os.path.normpath(sys.prefix))


# Ett register per (version, variant, cache-mapp). Manifestet är stort och tar
# tiotals sekunder att hämta första gången, så det återanvänds inom sessionen.
_client_cache = {}


def _client(dataset, cache_dir, messages=None):
    """
    GeoTessera-klient för en dataset-etikett, med tiles cachade i cache_dir.

    Första anropet hämtar manifestet över alla publicerade tiles. Det ligger kvar
    i geotesseras egen cache mellan körningar, men kostar en stund första gången
    och är värt ett meddelande i loggen.
    """
    _require_geotessera()
    if dataset not in DATASETS:
        raise ValueError("Okänd dataset-version: {}".format(dataset))
    version, variant = DATASETS[dataset]

    key = (version, variant, os.path.abspath(str(cache_dir)))
    if key in _client_cache:
        return _client_cache[key]

    if messages is not None:
        messages.addMessage(
            "Läser Tessera-registret ({} {})... första gången hämtas ett "
            "manifest över alla tiles, vilket tar en stund.".format(version, variant)
        )
    try:
        client = GeoTessera(dataset_version=version, dataset_variant=variant,
                            embeddings_dir=str(cache_dir))
    except Exception as exc:                                # noqa: BLE001
        raise ValueError(
            "Kunde inte läsa Tessera-registret för {} {}: {}".format(version, variant, exc)
        )
    _client_cache[key] = client
    return client


def _available_years(client):
    """År som finns i registret, som strängar."""
    try:
        return [str(y) for y in client.registry.get_available_years()]
    except Exception:                                       # noqa: BLE001
        return list(YEARS)


def _tile_sizes(client, year, tiles):
    """
    {(lon, lat): byte} för de tiles i listan som faktiskt finns, hämtat ur
    registret. Storleken omfattar embedding, scales och landmask, alltså allt
    verktyget laddar ned per tile. Registret gör att uppskattningen är exakt
    utan ett enda extra nätverksanrop.
    """
    wanted = set(tiles)
    if not wanted:
        return {}

    lons = [t[0] for t in wanted]
    lats = [t[1] for t in wanted]
    bounds = (min(lons) - TILE_DEG / 2, min(lats) - TILE_DEG / 2,
              max(lons) + TILE_DEG / 2, max(lats) + TILE_DEG / 2)

    sizes = {}
    try:
        found = list(client.registry.iter_tiles_in_region(bounds, int(year)))
    except Exception as exc:                                # noqa: BLE001
        raise ValueError("Kunde inte söka i Tessera-registret: {}".format(exc))

    for _year, lon, lat in found:
        tile = (round(float(lon), 2), round(float(lat), 2))
        if tile not in wanted or tile in sizes:
            continue
        total = 0
        for getter, args in (
            (client.registry.get_tile_file_size, (int(year), tile[0], tile[1])),
            (client.registry.get_scales_file_size, (int(year), tile[0], tile[1])),
            (client.registry.get_landmask_file_size, (tile[0], tile[1])),
        ):
            try:
                total += int(getter(*args) or 0)
            except Exception:                               # noqa: BLE001
                pass
        embedding = 0
        try:
            embedding = int(
                client.registry.get_tile_file_size(int(year), tile[0], tile[1]) or 0)
        except Exception:                                   # noqa: BLE001
            pass
        if not embedding:
            embedding = _approx_embedding_bytes(tile[1])
        sizes[tile] = (total or embedding, embedding)
    return sizes


def _fetch_tile_files(client, year, lon, lat):
    """
    (embedding, scales, landmask) som lokala sökvägar, nedladdade vid behov.
    geotessera återanvänder filer som redan ligger i cache-mappen.
    """
    embedding = client.registry.fetch(year=int(year), lon=lon, lat=lat, is_scales=False)
    scales = client.registry.fetch(year=int(year), lon=lon, lat=lat, is_scales=True)
    landmask = client.registry.fetch_landmask(lon=lon, lat=lat)
    return embedding, scales, landmask


# =============================================================================
# Storleksuppskattning
# =============================================================================

def _duration(seconds):
    """Sekunder som läsbar text: '42 s' eller '3 min 20 s'."""
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


def _approx_embedding_bytes(lat):
    """
    Ungefärlig storlek på en embedding-tile på en given latitud, som reserv när
    registret inte kan svara.

    En tile är 0,1 grader i 10 m-celler. Höjden är i praktiken ~1130 celler och
    bredden krymper med cos(lat). Faktorerna kommer från uppmätta tiles, som har
    en liten marginal utöver den nominella storleken.
    """
    height = TILE_DEG * 111_320.0 / CELL_SIZE_M * 1.02
    width = TILE_DEG * 111_320.0 * math.cos(math.radians(lat)) / CELL_SIZE_M * 1.07
    return int(max(height * width, 1.0) * N_BANDS) + NPY_HEADER_BYTES


def _approx_tile_bytes(lat):
    """Embedding plus scales plus landmask, ungefärligt."""
    embedding = _approx_embedding_bytes(lat)
    return embedding + _scales_bytes(embedding) + LANDMASK_BYTES


def _scales_bytes(embedding_bytes):
    """
    Storleken på tilens scales-fil, härledd ur embeddingens storlek.

    Embeddingen är (H, W, 128) int8 och scales (H, W) float32, båda med samma
    128-byte .npy-header, så H*W = (bytes - header) / 128.
    """
    return _tile_pixels(embedding_bytes) * 4 + NPY_HEADER_BYTES


def _tile_pixels(embedding_bytes):
    """Antal pixlar i en tile, ur embeddingfilens storlek."""
    return max((int(embedding_bytes) - NPY_HEADER_BYTES) // N_BANDS, 0)


def _download_bytes(sizes):
    """Totalt att hämta för {(lon, lat): (nedladdning, embedding)}."""
    return sum(int(d) for d, _e in sizes.values())


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
        pixels = sum(_tile_pixels(e) for _d, e in sizes.values())
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
    per_tile = [_tile_pixels(e) * n_bands * 4 for _d, e in sizes.values()]
    if not per_tile:
        return 0
    return int(sum(per_tile) + max(per_tile))


# =============================================================================
# Bounding box
# =============================================================================

_SOURCE_OWN = "utbredningens eget koordinatsystem"
_SOURCE_PARAM = "parametern 'Koordinatsystem för utbredningen'"
_SOURCE_MAP = "den aktiva kartans koordinatsystem"
_SOURCE_NONE = "ingen aktiv karta, så SWEREF 99 TM antas"


def _extent_fallback(crs_spec):
    """
    (koordinatsystem, källa) för en utbredning som saknar eget: parametern om
    den är ifylld, annars den aktiva kartans, annars SWEREF 99 TM.
    """
    map_sr = _map_sr()
    sr = _coerce_sr(crs_spec)
    if sr is not None:
        # Parameterns standardvärde är kartans system; säg då det.
        if map_sr is not None and _sr_label(sr) == _sr_label(map_sr):
            return sr, _SOURCE_MAP
        return sr, _SOURCE_PARAM
    if map_sr is not None:
        return map_sr, _SOURCE_MAP
    return _sr(SWEREF99TM_WKID), _SOURCE_NONE


def _extent_from_value(value, text, fallback):
    """
    (arcpy.Extent, källa) ur en GPExtent-parameter, eller (None, None).

    Värdet är ett "geoprocessing extent object" (eller i tester ett Extent eller
    en sträng "xmin ymin xmax ymax"). Hörnen är vanliga tal. Koordinatsystemet
    följer bara med när utbredningen kommer från ett lager eller en datakälla,
    och då som WKT2 i valueAsText efter de fyra talen. Inskrivna koordinater och
    kartvyns utbredning har inget; då används fallback = (sr, källa).
    """
    if value is None and not text:
        return None, None

    corners, sr = None, None
    if value is not None and not isinstance(value, str):
        try:
            corners = [float(getattr(value, name))
                       for name in ("XMin", "YMin", "XMax", "YMax")]
        except (AttributeError, TypeError, ValueError):
            corners = None
        # Ett riktigt arcpy.Extent kan bära sitt koordinatsystem; GP-objektet
        # har spatialReference None.
        sr = _coerce_sr(getattr(value, "spatialReference", None))

    raw = text if text else (value if isinstance(value, str) else "")
    parts = (raw or "").strip().split(None, 4)
    if corners is None:
        try:
            corners = [float(p.replace(",", ".")) for p in parts[:4]]
        except ValueError:
            return None, None
        if len(corners) < 4:
            return None, None
    if sr is None and len(parts) == 5:
        sr = _coerce_sr(parts[4])

    # Utbredningen byggs alltid om till ett riktigt arcpy.Extent. GP-objektets
    # .polygon ger en geometri vars projectAs misslyckas med "CreateObject error
    # creating spatial reference".
    if sr is not None:
        source = _SOURCE_OWN
    else:
        sr, source = fallback if fallback else _extent_fallback(None)
    return arcpy.Extent(corners[0], corners[1], corners[2], corners[3],
                        spatial_reference=sr), source


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
            "Utbredningen saknar koordinatsystem. Ange ett under "
            "'Koordinatsystem för utbredningen'."
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
# Polygoner
# =============================================================================

def _normalize_aoi_mode(text):
    """Valt avgränsningssätt, med den gamla polygonetiketten som alias."""
    text = (text or "").strip()
    if text in (AOI_POLYGONS, _AOI_POLYGONS_OLD):
        return AOI_POLYGONS
    if text == AOI_EXTENT:
        return AOI_EXTENT
    return text or DEFAULT_AOI_MODE


def _polygon_schema():
    """
    Tom polygonfeatureklass i SWEREF 99 TM, standardvärde för Feature Set-
    parametern så att dialogens ritverktyg ritar polygoner.

    Namnet görs unikt i stället för Exists/Delete på ett fast namn: memory-
    arbetsytan delas av hela Pro-sessionen, och Delete där har fallerat med
    "Invalid SQL syntax ... GDB_Items".
    """
    name = "aoi_schema_{}".format(uuid.uuid4().hex[:12])
    try:
        arcpy.management.CreateFeatureclass(
            "memory", name, "POLYGON", spatial_reference=_sr(SWEREF99TM_WKID))
    except Exception:                                       # noqa: BLE001
        return None
    return "memory/" + name


_NO_POLYGONS = "Rita minst en polygon i kartan eller välj ett polygonlager."


def _densify(geom):
    """
    Förtäta kanterna innan omprojicering, så att långa raka kanter följer med
    när de böjs i det nya systemet. Avståndet är en femtiondel av geometrins
    största sida och därmed i geometrins egna enheter.
    """
    try:
        ext = geom.extent
        span = max(ext.width, ext.height)
        if span > 0:
            return geom.densify("DISTANCE", span / 50.0)
    except Exception:                                       # noqa: BLE001
        pass
    return geom


def _project_poly(geom, target_sr):
    """Förtätad och omprojicerad geometri, oförändrad om systemet redan stämmer."""
    try:
        source = geom.spatialReference
        if source is not None and source.factoryCode and \
                source.factoryCode == target_sr.factoryCode:
            return geom
    except Exception:                                       # noqa: BLE001
        pass
    return _project_geometry(_densify(geom), target_sr)


def _union_all(geoms):
    """Slå ihop geometrier parvis, vilket är snabbare än en lång kedja för många objekt."""
    geoms = list(geoms)
    while len(geoms) > 1:
        merged = []
        for i in range(0, len(geoms), 2):
            if i + 1 < len(geoms):
                merged.append(geoms[i].union(geoms[i + 1]))
            else:
                merged.append(geoms[i])
        geoms = merged
    return geoms[0] if geoms else None


def _aoi_polygons(value):
    """
    (sammanslagen polygon, antal objekt) ur polygonparametern, i polygonernas
    eget koordinatsystem.

    Värdet kan vara ett lager (då används bara de valda objekten om det finns ett
    urval), ritade objekt (ett record set) eller en sökväg. SearchCursor och
    Describe fungerar på alla tre.
    """
    if value is None:
        raise ValueError(_NO_POLYGONS)
    try:
        sr_in = arcpy.Describe(value).spatialReference
    except Exception as exc:                                # noqa: BLE001
        raise ValueError("Kunde inte läsa polygonerna: {}".format(exc))
    # projectAs på en geometri utan koordinatsystem ger tillbaka indata oförändrat,
    # vilket skulle hämta data för fel plats utan fel. Kontrollera därför först.
    if not _sr_is_valid(sr_in):
        raise ValueError(
            "Polygonerna saknar koordinatsystem. Definiera ett för lagret "
            "(Define Projection) och kör igen.")

    shapes = []
    with arcpy.da.SearchCursor(value, ["SHAPE@"], spatial_reference=sr_in) as cursor:
        for (shape,) in cursor:
            if shape is None or not shape.area or shape.area <= 0:
                continue
            shapes.append(shape)
    if not shapes:
        raise ValueError(_NO_POLYGONS)
    return _union_all(shapes), len(shapes)


def _tile_rect_wgs84(lon, lat):
    """Tilens nominella cell (0,1 x 0,1 grader) som polygon i WGS84."""
    half = TILE_DEG / 2.0
    return arcpy.Polygon(arcpy.Array([
        arcpy.Point(lon - half, lat - half), arcpy.Point(lon - half, lat + half),
        arcpy.Point(lon + half, lat + half), arcpy.Point(lon + half, lat - half),
        arcpy.Point(lon - half, lat - half),
    ]), _sr(WGS84_WKID))


def _tiles_for_polygon(poly_wgs84):
    """
    (tiles som skär polygonen, antal tiles inom polygonens bounding box).

    Tilarnas kanter är räta linjer i WGS84, så testet görs där mot polygonen
    omprojicerad dit (förtätad, se _project_poly).
    """
    ext = poly_wgs84.extent
    candidates = _tiles_for_bbox(ext.XMin, ext.YMin, ext.XMax, ext.YMax)
    hits = [t for t in candidates
            if not poly_wgs84.disjoint(_tile_rect_wgs84(t[0], t[1]))]
    return hits, len(candidates)


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
    Standardmapp för nedladdade tiles, skapad om den saknas.

    Inte tempfile.gettempdir(): inne i Pro pekar den på en eget mapp per session
    (ArcGISProTemp<nnnn>) som dessutom ofta blir kvar när Pro stängs. En sådan
    sökväg finns inte förrän någon skapar den, vilket gör att parametern faller
    på ERROR 000732 redan när dialogen öppnas, och den byter namn varje gång Pro
    startas om så att ingenting någonsin återanvänds.

    Inte heller projektmappen — den ligger ofta i OneDrive, och rådata för ett
    par tiles är hundratals MB som då skulle synkas till molnet.

    Mappen skapas här, inte vid körning, just för att parametern ska validera.
    """
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    path = os.path.join(base, _CACHE_DIRNAME)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def _write_tool_metadata(tool_cls, toolbox_alias):
    """
    Skriv verktygets metadatafil med parameterförklaringar från TOOLTIPS.

    Pro läser verktygstipsen i dialogen från <verktygslåda>.<verktyg>.pyt.xml
    (elementet dialogReference per parameter). Det finns inget attribut på
    arcpy.Parameter för detta. Filen skrivs bara om innehållet har ändrats.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    toolbox = os.path.splitext(os.path.basename(__file__))[0]
    path = os.path.join(here, "{}.{}.pyt.xml".format(toolbox, tool_cls.__name__))

    def html(text):
        body = escape(text).replace("\n", "</SPAN></P><P><SPAN>")
        return escape('<DIV STYLE="text-align:Left;"><P><SPAN>{}</SPAN></P></DIV>'.format(body))

    tool = tool_cls()
    params = []
    for p in tool.getParameterInfo():
        tip = TOOLTIPS.get(p.name)
        if not tip:
            continue
        params.append(
            '<param name="{n}" displayname="{d}" type="{t}" direction="{r}">'
            "<dialogReference>{h}</dialogReference>"
            "<pythonReference>{h}</pythonReference></param>".format(
                n=p.name, d=escape(p.displayName, {'"': "&quot;"}),
                t=p.parameterType, r=p.direction, h=html(tip))
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<metadata xml:lang="sv"><Esri><ArcGISFormat>1.0</ArcGISFormat></Esri>'
        '<tool name="{name}" displayname="{label}" toolboxalias="{alias}" xmlns="">'
        "<parameters>{params}</parameters><summary>{summary}</summary></tool>"
        "<dataIdInfo><idCitation><resTitle>{label}</resTitle></idCitation>"
        "<idAbs>{summary}</idAbs></dataIdInfo></metadata>\n"
    ).format(name=tool_cls.__name__, label=escape(tool.label), alias=toolbox_alias,
             params="".join(params), summary=html(TOOL_SUMMARY))

    try:
        with open(path, encoding="utf-8") as fh:
            if fh.read() == xml:
                return
    except OSError:
        pass
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(xml)
    except OSError:
        # Skrivskyddad plats: verktyget fungerar ändå, bara utan verktygstips.
        pass


# =============================================================================
# Toolbox
# =============================================================================

class Toolbox:
    def __init__(self):
        self.label = "Tessera"
        self.alias = "tessera"
        self.tools = [HamtaTesseraRaster]
        _write_tool_metadata(HamtaTesseraRaster, self.alias)


class HamtaTesseraRaster:
    def __init__(self):
        self.label = "Hämta Tessera-raster till geodatabas"
        self.description = TOOL_SUMMARY + (
            "\n\nData hämtas kvantiserat som int8 med en skalfaktor per pixel och skrivs "
            "dekvantiserat som 32-bitars float."
        )
        self.canRunInBackground = False
        # Senaste uppskattningen, så att den inte räknas om när användaren ändrar
        # något som inte påverkar storleken (t.ex. skriver rasternamnet).
        self._estimate_memo = (None, "")
        # Senast föreslagna rasternamnet, se updateParameters.
        self._name_memo = "tessera_{}".format(DEFAULT_YEAR)
        self._cache_dir = _default_cache_dir()
        # Utbredningen för ett lager med urval, som kräver en cursor. Nyckel:
        # (lagrets text, urvalet).
        self._bbox_memo = (None, None)

    # ── Parametrar ────────────────────────────────────────────────────────────

    def getParameterInfo(self):
        p_aoi_mode = arcpy.Parameter(
            displayName="Avgränsa området med",
            name="aoi_mode", datatype="GPString",
            parameterType="Required", direction="Input",
        )
        p_aoi_mode.filter.type = "ValueList"
        p_aoi_mode.filter.list = [AOI_POLYGONS, AOI_EXTENT]
        p_aoi_mode.value = DEFAULT_AOI_MODE

        # Polygoner och utbredning är båda Optional i ramverket; updateMessages
        # kräver den som valts.
        p_aoi = arcpy.Parameter(
            displayName="Intresseområde (polygoner)",
            name="aoi", datatype="GPFeatureRecordSetLayer",
            parameterType="Optional", direction="Input",
        )
        p_aoi.filter.list = ["Polygon"]
        # Ett tomt polygonschema som standard gör att dialogens ritverktyg ritar
        # polygoner. Följden är att "inget valt" inte är None utan noll objekt.
        schema = _polygon_schema()
        if schema:
            p_aoi.value = schema
        p_aoi.enabled = DEFAULT_AOI_MODE == AOI_POLYGONS

        p_extent = arcpy.Parameter(
            displayName="Utbredning",
            name="extent", datatype="GPExtent",
            parameterType="Optional", direction="Input",
        )
        p_extent.value = _default_extent()
        p_extent.enabled = DEFAULT_AOI_MODE == AOI_EXTENT

        p_extent_crs = arcpy.Parameter(
            displayName="Koordinatsystem för utbredningen (används om den saknar ett)",
            name="extent_crs", datatype="GPCoordinateSystem",
            parameterType="Optional", direction="Input",
        )
        p_extent_crs.value = _default_extent_crs()
        p_extent_crs.enabled = DEFAULT_AOI_MODE == AOI_EXTENT

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
            displayName="Behåll nedladdade tiles efter körningen (snabbare omkörning, "
                        "hundratals MB per tile)",
            name="keep_cache", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Nedladdning",
        )
        p_keep.value = False

        p_add = arcpy.Parameter(
            displayName="Lägg till resultatet i kartan",
            name="add_to_map", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Karta",
        )
        p_add.value = True

        return [p_aoi_mode, p_aoi, p_extent, p_extent_crs, p_year, p_dataset, p_bands,
                p_estimate, p_exact, p_gdb, p_name, p_mode, p_target_crs, p_overwrite,
                p_max_gb, p_cache, p_keep, p_add]

    def isLicensed(self):
        return True

    # ── Dialog ────────────────────────────────────────────────────────────────

    def updateParameters(self, parameters):
        p = {q.name: q for q in parameters}

        # Den gamla etiketten från äldre skript byts mot den nya, så att
        # värdelistan godkänner den.
        if p["aoi_mode"].valueAsText == _AOI_POLYGONS_OLD:
            p["aoi_mode"].value = AOI_POLYGONS
        by_polygons = _normalize_aoi_mode(p["aoi_mode"].valueAsText) == AOI_POLYGONS
        p["aoi"].enabled = by_polygons
        p["extent"].enabled = not by_polygons
        # Polygonerna har sitt eget koordinatsystem; parametern gäller bara rutan.
        p["extent_crs"].enabled = not by_polygons

        # Namnförslaget följer valt år tills användaren skrivit ett eget namn.
        # Jämförelsen görs mot det senast föreslagna namnet i stället för mot
        # parameterns altered-flagga, som också sätts när koden själv skriver
        # värdet här.
        suggestion = "tessera_{}".format(p["year"].valueAsText or DEFAULT_YEAR)
        current = (p["out_name"].valueAsText or "").strip()
        if current in ("", self._name_memo):
            p["out_name"].value = suggestion
        self._name_memo = suggestion

        # Cache-mappen styr var geotessera lägger sina filer och behövs när
        # uppskattningen öppnar registret.
        self._cache_dir = p["cache_dir"].valueAsText or _default_cache_dir()

        # Målkoordinatsystemet gäller bara mosaiken.
        p["target_crs"].enabled = (p["out_mode"].valueAsText == MODE_MOSAIC)

        # Uppskattningen är ett rent utdatafält. enabled sätts även här, inte
        # bara i getParameterInfo, eftersom Pro återställer flaggan när dialogen
        # laddas om.
        p["estimate"].enabled = False

        ext, error = self._area_extent(p, by_polygons)
        area_key = error if ext is None else (
            round(ext.XMin, 3), round(ext.YMin, 3), round(ext.XMax, 3), round(ext.YMax, 3),
            _sr_label(ext.spatialReference))
        key = (
            str(self._cache_dir), by_polygons, area_key,
            p["year"].valueAsText, p["dataset"].valueAsText, p["bands"].valueAsText,
            p["out_mode"].valueAsText, str(p["target_crs"].valueAsText),
            bool(p["exact_estimate"].value),
        )
        if key != self._estimate_memo[0]:
            self._estimate_memo = (key, self._estimate_text(
                ext, error, by_polygons, p["year"], p["dataset"], p["bands"],
                p["out_mode"], p["target_crs"], exact=bool(p["exact_estimate"].value),
            ))
        p["estimate"].value = self._estimate_memo[1]

    def _area_extent(self, p, by_polygons):
        """
        (Extent med koordinatsystem, None) för områdets rektangel, eller
        (None, feltext). Med polygoner är det deras bounding box. Billig nog för
        varje anrop av updateParameters.
        """
        if by_polygons:
            return self._polygon_bbox(p["aoi"])
        fallback = _extent_fallback(self._param_sr(p["extent_crs"]))
        ext, _source = _extent_from_value(p["extent"].value, p["extent"].valueAsText,
                                          fallback)
        if ext is None:
            return None, "Ange en utbredning."
        return ext, None

    def _polygon_bbox(self, p_aoi):
        """
        Polygonernas bounding box utan att läsa alla objekt.

        Describe ger utbredningen direkt ur datakällan, men den bortser från ett
        urval i lagret. Bara när lagret har ett urval läses de valda objektens
        utbredningar med en cursor, och resultatet sparas tills urvalet ändras.
        Ett tomt schema (inget ritat) har utbredningen NaN.
        """
        value = p_aoi.value
        if value is None or not p_aoi.valueAsText:
            return None, _NO_POLYGONS
        try:
            desc = arcpy.Describe(value)
            sr = desc.spatialReference
        except Exception as exc:                            # noqa: BLE001
            return None, "Kunde inte läsa polygonerna: {}".format(exc)
        if not _sr_is_valid(sr):
            return None, "Polygonerna saknar koordinatsystem."

        try:
            fids = desc.FIDSet or ""
        except Exception:                                   # noqa: BLE001
            fids = ""
        if fids:
            key = (p_aoi.valueAsText, hash(fids))
            if self._bbox_memo[0] != key:
                self._bbox_memo = (key, _selected_bbox(value, sr))
            return self._bbox_memo[1]

        try:
            e = desc.extent
            corners = (e.XMin, e.YMin, e.XMax, e.YMax)
        except Exception:                                   # noqa: BLE001
            corners = None
        if corners is None or any(c is None or math.isnan(c) for c in corners):
            return None, _NO_POLYGONS
        return arcpy.Extent(corners[0], corners[1], corners[2], corners[3],
                            spatial_reference=sr), None

    def _estimate_text(self, ext, error, by_polygons, p_year, p_dataset, p_bands,
                       p_mode, p_target_crs, exact):
        """
        Text till fältet "Uppskattad storlek".

        Utan exakt kontroll räknas storleken analytiskt ur antalet tiles, vilket
        är omedelbart men förutsätter att alla tiles finns — tiles över öppet
        vatten publiceras inte. Med exakt kontroll frågas servern om varje tile.
        Med polygoner räknas på deras bounding box, vilket ger ett tak.
        """
        try:
            if ext is None:
                return error or "Ange ett område."

            bbox = _project_extent(ext, _sr(WGS84_WKID))
            tiles = _tiles_for_bbox(bbox.XMin, bbox.YMin, bbox.XMax, bbox.YMax)
            if not tiles:
                return "Området täcker inga tiles."

            band_indices = _parse_bands(p_bands.valueAsText)
            mode = p_mode.valueAsText or MODE_MOSAIC

            bbox_target = None
            if mode == MODE_MOSAIC:
                target_sr = self._target_sr(p_target_crs)
                target_ext = _snap_extent(_project_extent(ext, target_sr))
                bbox_target = (target_ext.XMin, target_ext.YMin,
                               target_ext.XMax, target_ext.YMax)

            where = " inom polygonernas bounding box" if by_polygons else ""
            if exact:
                client = _client(p_dataset.valueAsText or DEFAULT_DATASET,
                                 self._cache_dir_value())
                available = _tile_sizes(client, p_year.valueAsText or DEFAULT_YEAR, tiles)
                if not available:
                    return ("Området saknar publicerad data för {} — ingen av "
                            "{} tiles finns.".format(p_year.valueAsText, len(tiles)))
                prefix = "{} av {} tiles{} finns".format(len(available), len(tiles), where)
            else:
                available = {t: (_approx_tile_bytes(t[1]),
                                 _approx_embedding_bytes(t[1])) for t in tiles}
                prefix = "{} {} tiles{} (förutsätter att alla finns)".format(
                    "högst" if by_polygons else "ca", len(tiles), where)

            download = _download_bytes(available)
            in_gdb = _gdb_bytes(available, len(band_indices), mode, bbox_target)
            scratch = _scratch_bytes(available, len(band_indices), mode)
            if by_polygons and mode == MODE_MOSAIC:
                # Den oklippta mosaiken ligger i geodatabasen tills klippningen är klar.
                scratch += in_gdb

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

    def _cache_dir_value(self):
        return getattr(self, "_cache_dir", None) or _default_cache_dir()

    @staticmethod
    def _param_sr(parameter):
        """
        Koordinatsystemet ur en GPCoordinateSystem-parameter, eller None om den
        är tom. Egna koordinatsystem (faktorkod 0 men med WKT-definition)
        behålls — annars skulle koordinaterna tystlåtet tolkas som något annat.
        """
        if parameter.value is None:
            return None
        return _coerce_sr(parameter.valueAsText) or _coerce_sr(parameter.value)

    @classmethod
    def _target_sr(cls, p_target_crs):
        return cls._param_sr(p_target_crs) or _sr(SWEREF99TM_WKID)

    def updateMessages(self, parameters):
        p = {q.name: q for q in parameters}
        by_polygons = _normalize_aoi_mode(p["aoi_mode"].valueAsText) == AOI_POLYGONS

        try:
            _parse_bands(p["bands"].valueAsText)
        except ValueError as exc:
            p["bands"].setErrorMessage(str(exc))

        gdb = p["out_gdb"].valueAsText
        if gdb and not gdb.lower().rstrip("\\/").endswith(".gdb"):
            p["out_gdb"].setErrorMessage("Utdata måste vara en filgeodatabas (.gdb).")

        name = (p["out_name"].valueAsText or "").strip()
        if name and not re.fullmatch(r"[A-Za-zÅÄÖåäö_][A-Za-zÅÄÖåäö0-9_]*", name):
            p["out_name"].setErrorMessage(
                "Rasternamnet får bara innehålla bokstäver, siffror och understreck, "
                "och måste börja med en bokstav eller ett understreck."
            )

        # Polygoner och utbredning är Optional i ramverket; den valda krävs här.
        if by_polygons:
            area_param = p["aoi"]
            ext, error = self._polygon_bbox(area_param)
            if ext is None:
                area_param.setErrorMessage(error)
        else:
            area_param = p["extent"]
            fallback = _extent_fallback(self._param_sr(p["extent_crs"]))
            ext, source = _extent_from_value(area_param.value, area_param.valueAsText,
                                             fallback)
            if ext is None:
                area_param.setErrorMessage("Ange en utbredning.")
            elif ext.XMax <= ext.XMin or ext.YMax <= ext.YMin:
                area_param.setErrorMessage("Utbredningen har ingen yta.")
                ext = None
            elif source == _SOURCE_PARAM:
                # Rutan kommer från kartan men bär inget koordinatsystem med sig,
                # så en avvikelse här betyder nästan alltid fel plats.
                map_sr = _map_sr()
                if map_sr is not None and _sr_label(map_sr) != _sr_label(ext.spatialReference):
                    p["extent_crs"].setWarningMessage(
                        "Kartan använder {}. Rutan tolkas som {} — kontrollera att det är "
                        "rätt, annars hämtas data för fel plats.".format(
                            _sr_label(map_sr), _sr_label(ext.spatialReference))
                    )

        if ext is not None:
            try:
                metric = _project_extent(
                    ext,
                    self._target_sr(p["target_crs"])
                    if p["out_mode"].valueAsText == MODE_MOSAIC else _sr(SWEREF99TM_WKID),
                )
                if max(metric.width, metric.height) > _MAX_SANE_SIDE_M:
                    area_param.setWarningMessage(
                        "Området är över {:.0f} km på en sida. Tessera är ca 90 MB per "
                        "tile om 11 km — kontrollera storleksuppskattningen innan du "
                        "kör.".format(_MAX_SANE_SIDE_M / 1000.0)
                    )
            except Exception:
                pass

        if p["max_gb"].value is not None and p["max_gb"].value <= 0:
            p["max_gb"].setErrorMessage("Gränsen måste vara större än noll.")

        cache = p["cache_dir"].valueAsText
        if cache and any(hint in cache.lower() for hint in _SYNC_HINTS):
            p["cache_dir"].setWarningMessage(
                "Mappen ser ut att synkas till molnet. Nedladdade tiles är hundratals "
                "MB — välj hellre en lokal mapp, t.ex. {}.".format(_default_cache_dir())
            )

    # ── Körning ───────────────────────────────────────────────────────────────

    def execute(self, parameters, messages):
        p = {q.name: q for q in parameters}
        aoi_mode = _normalize_aoi_mode(p["aoi_mode"].valueAsText)
        by_polygons = aoi_mode == AOI_POLYGONS

        error = None
        try:
            _run(
                aoi_mode=aoi_mode,
                polygons=p["aoi"].value if by_polygons else None,
                extent_value=None if by_polygons else p["extent"].value,
                extent_text=None if by_polygons else p["extent"].valueAsText,
                extent_crs=self._param_sr(p["extent_crs"]),
                year=p["year"].valueAsText or DEFAULT_YEAR,
                dataset=p["dataset"].valueAsText or DEFAULT_DATASET,
                bands_text=p["bands"].valueAsText,
                out_gdb=p["out_gdb"].valueAsText,
                out_name=(p["out_name"].valueAsText or "").strip(),
                mode=p["out_mode"].valueAsText or MODE_MOSAIC,
                target_sr=self._target_sr(p["target_crs"]),
                overwrite=bool(p["overwrite"].value),
                max_gb=float(p["max_gb"].value) if p["max_gb"].value else 0.0,
                cache_dir=p["cache_dir"].valueAsText or _default_cache_dir(),
                keep_cache=bool(p["keep_cache"].value),
                add_to_map=bool(p["add_to_map"].value),
                messages=messages,
            )
        except ValueError as exc:
            error = str(exc)
        except OSError as exc:
            error = "Nätverks- eller filfel vid hämtning från Tessera: {}".format(exc)
        # ExecuteError höjs utanför except-blocket; inifrån skriver GP-loggen ut
        # hela den kedjade tracebacken under det läsbara meddelandet.
        if error is not None:
            messages.addErrorMessage(error)
            raise arcpy.ExecuteError

    def postExecute(self, parameters):
        return


def _selected_bbox(value, sr):
    """(Extent, None) för de valda objekten i ett lager, eller (None, feltext)."""
    xs, ys = [], []
    try:
        with arcpy.da.SearchCursor(value, ["SHAPE@"], spatial_reference=sr) as cursor:
            for (shape,) in cursor:
                if shape is None:
                    continue
                e = shape.extent
                xs += [e.XMin, e.XMax]
                ys += [e.YMin, e.YMax]
    except Exception as exc:                                # noqa: BLE001
        return None, "Kunde inte läsa polygonerna: {}".format(exc)
    if not xs:
        return None, _NO_POLYGONS
    return arcpy.Extent(min(xs), min(ys), max(xs), max(ys), spatial_reference=sr), None


# =============================================================================
# Körningens innehåll (separat funktion — går att testa utanför Pro)
# =============================================================================

def _run(aoi_mode, polygons, extent_value, extent_text, extent_crs, year, dataset,
         bands_text, out_gdb, out_name, mode, target_sr, overwrite, max_gb, cache_dir,
         keep_cache, add_to_map, messages):
    """
    Utför hela hämtningen. Returnerar listan med skapade rasterdataset.

    aoi_mode väljer om polygons (lager, record set eller sökväg) eller
    extent_value/extent_text (GPExtent-värdet och dess valueAsText) avgränsar
    området. extent_crs används för en utbredning som saknar eget
    koordinatsystem; None betyder den aktiva kartans, och utan karta SWEREF 99 TM.
    """

    _require_geotessera()
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

    # 1. Område och tiles
    arcpy.SetProgressorLabel("Läser området...")
    by_polygons = _normalize_aoi_mode(aoi_mode) == AOI_POLYGONS
    clip_geom = None
    if by_polygons:
        geom, count = _aoi_polygons(polygons)
        messages.addMessage(
            "Området: {} polygon{} i {}, sammanlagt {:.2f} km² (polygonernas eget "
            "koordinatsystem).".format(
                count, "" if count == 1 else "er", _sr_label(geom.spatialReference),
                _area_km2(geom))
        )
        poly_wgs = _project_poly(geom, _sr(WGS84_WKID))
        tiles, in_bbox = _tiles_for_polygon(poly_wgs)
        bbox = poly_wgs.extent
        messages.addMessage(
            "Polygonernas bounding box (WGS84): {:.4f}, {:.4f} — {:.4f}, {:.4f}".format(
                bbox.XMin, bbox.YMin, bbox.XMax, bbox.YMax)
        )
        messages.addMessage(
            "{} tiles skär polygonerna ({} inom deras bounding box; {} band per pixel, "
            "{}).".format(len(tiles), in_bbox, len(band_indices), dataset)
        )
        if mode == MODE_MOSAIC:
            clip_geom = _project_poly(geom, target_sr)
            target_ext = _snap_extent(clip_geom.extent)
        else:
            target_ext = None
    else:
        ext, source = _extent_from_value(extent_value, extent_text,
                                         _extent_fallback(extent_crs))
        if ext is None:
            raise ValueError("Ange en utbredning.")
        if ext.XMax <= ext.XMin or ext.YMax <= ext.YMin:
            raise ValueError("Utbredningen har ingen yta.")

        # Vilket koordinatsystem rutan tolkas i skrivs ut. Inskrivna koordinater
        # och kartvyns utbredning är bara fyra tal, så tolkningen är ett antagande:
        # syns den i loggen går det att upptäcka att data hämtats för fel plats.
        messages.addMessage(
            "Utbredningen tolkas som {} ({}): {:.2f}, {:.2f} till {:.2f}, {:.2f}".format(
                _sr_label(ext.spatialReference), source,
                ext.XMin, ext.YMin, ext.XMax, ext.YMax)
        )
        map_sr = _map_sr()
        if source != _SOURCE_OWN and map_sr is not None and \
                _sr_label(map_sr) != _sr_label(ext.spatialReference):
            messages.addWarningMessage(
                "Kartan använder {} men rutan tolkas som {}. Stämmer det inte hämtas data "
                "för fel plats — ändra 'Koordinatsystem för utbredningen'.".format(
                    _sr_label(map_sr), _sr_label(ext.spatialReference))
            )

        bbox = _project_extent(ext, _sr(WGS84_WKID))
        tiles = _tiles_for_bbox(bbox.XMin, bbox.YMin, bbox.XMax, bbox.YMax)
        messages.addMessage(
            "Utbredningen i WGS84: {:.4f}, {:.4f} — {:.4f}, {:.4f}".format(
                bbox.XMin, bbox.YMin, bbox.XMax, bbox.YMax)
        )
        messages.addMessage(
            "{} tiles i rutan ({} band per pixel, {}).".format(
                len(tiles), len(band_indices), dataset)
        )
        target_ext = (_snap_extent(_project_extent(ext, target_sr))
                      if mode == MODE_MOSAIC else None)

    if not tiles:
        raise ValueError("Området täcker inga tiles.")

    # 2. Storlek — alltid innan något hämtas
    arcpy.SetProgressorLabel("Läser Tessera-registret...")
    client = _client(dataset, cache_dir, messages)
    sizes = _tile_sizes(client, year, tiles)
    available = sizes
    if not available:
        raise ValueError(
            "Området saknar publicerad data för {} i {} — ingen av områdets {} tiles "
            "finns. Tiles över öppet vatten publiceras inte, och alla år finns inte "
            "i alla dataset-versioner.".format(year, dataset, len(tiles))
        )
    if len(available) < len(tiles):
        messages.addWarningMessage(
            "{} av {} tiles saknas för {} och hoppas över (öppet vatten eller ingen "
            "täckning).".format(len(tiles) - len(available), len(tiles), year)
        )

    bbox_target = None
    if target_ext is not None:
        bbox_target = (target_ext.XMin, target_ext.YMin, target_ext.XMax, target_ext.YMax)

    download_total = _download_bytes(available)
    gdb_total = _gdb_bytes(available, len(band_indices), mode, bbox_target)
    scratch_total = _scratch_bytes(available, len(band_indices), mode)
    # Med polygoner skrivs mosaiken först oklippt till geodatabasen och klipps
    # sedan till sitt slutliga namn, så geodatabasen behöver dubbelt så mycket
    # en stund.
    gdb_temp = gdb_total if clip_geom is not None else 0

    messages.addMessage("Att hämta      : {} ({} tiles)".format(
        _human_size(download_total), len(available)))
    messages.addMessage("I geodatabasen : ca {}".format(_human_size(gdb_total)))
    if scratch_total:
        messages.addMessage("Tillfälligt    : ca {} i arbetsmappen".format(
            _human_size(scratch_total)))
    if gdb_temp:
        messages.addMessage("Tillfälligt    : ca {} i geodatabasen (oklippt mosaik)".format(
            _human_size(gdb_temp)))

    if max_gb and download_total > max_gb * 1024 ** 3:
        raise ValueError(
            "Nedladdningen är {} och överstiger gränsen på {:.1f} GB. Minska området "
            "eller höj gränsen under 'Nedladdning'.".format(
                _human_size(download_total), max_gb)
        )

    _check_disk_space(cache_dir, download_total, gdb_total + gdb_temp, scratch_total,
                      out_gdb, mode, messages)

    # 3. Hämta rådata
    started = time.time()
    cached_bytes = _fetch_tiles(client, available, year, messages)
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
        outputs = _build_mosaic(client, available, year, band_indices,
                                out_gdb, out_name, target_sr, target_ext,
                                overwrite, messages, clip_geom)
    else:
        if by_polygons:
            messages.addMessage(
                "Tile-läge: de {} tiles som skär polygonerna skrivs hela, utan "
                "klippning.".format(len(available)))
        outputs = _build_tiles(client, available, year, band_indices,
                               out_gdb, out_name, overwrite, messages)
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
        _clear_cache(client, available, year, messages)

    messages.addMessage("Klar!")
    return outputs


def _area_km2(geom):
    """Ytan i km², geodetiskt så att den stämmer även för polygoner i grader."""
    try:
        return geom.getArea("GEODESIC", "SQUAREKILOMETERS")
    except Exception:                                       # noqa: BLE001
        return 0.0


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


def _fetch_tiles(client, sizes, year, messages):
    """
    Hämta embedding, scales och landmask för varje tile till cachen.
    Returnerar antalet byte som faktiskt hämtades.

    geotessera hämtar en fil i taget och hoppar över det som redan finns i
    cache-mappen, så en avbruten körning fortsätter där den slutade.
    """
    total = _download_bytes(sizes)
    done = 0
    fetched = 0
    percent = -1

    arcpy.SetProgressor("step", "Hämtar Tessera-tiles...", 0, 100, 1)
    try:
        for index, (tile, size) in enumerate(sorted(sizes.items()), start=1):
            lon, lat = tile
            tile_bytes = size[0] if isinstance(size, tuple) else size
            arcpy.SetProgressorLabel(
                "Hämtar {} ({}/{})".format(_grid_name(lon, lat), index, len(sizes))
            )
            before = _cached_bytes(client, year, lon, lat)
            try:
                _fetch_tile_files(client, year, lon, lat)
            except Exception as exc:                        # noqa: BLE001
                raise ValueError(
                    "Kunde inte hämta {}: {}".format(_grid_name(lon, lat), exc)
                )
            fetched += max(_cached_bytes(client, year, lon, lat) - before, 0)

            done += tile_bytes
            if total:
                # Förloppet uppdateras per tile och bara när procenten ändras;
                # SetProgressorPosition går via Pros gränssnitt och är inte gratis.
                new_percent = min(int(done * 100 / total), 100)
                if new_percent != percent:
                    percent = new_percent
                    arcpy.SetProgressorPosition(percent)
    finally:
        arcpy.ResetProgressor()

    return fetched


def _cached_bytes(client, year, lon, lat):
    """Hur mycket av tilen som redan ligger i cache-mappen."""
    total = 0
    for path in _cached_paths(client, year, lon, lat):
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def _cached_paths(client, year, lon, lat):
    """Förväntade sökvägar i cachen, utan att något hämtas."""
    name = _grid_name(lon, lat)
    root = str(getattr(client, "embeddings_dir", "") or "")
    if not root:
        return []
    emb_dir = os.path.join(root, gt_registry.EMBEDDINGS_DIR_NAME, str(year), name)
    return [
        os.path.join(emb_dir, name + ".npy"),
        os.path.join(emb_dir, name + "_scales.npy"),
        os.path.join(root, gt_registry.LANDMASKS_DIR_NAME, name + ".tiff"),
    ]


def _build_tiles(client, sizes, year, band_indices, out_gdb,
                 out_name, overwrite, messages):
    """En raster per tile i tilens egen UTM-projektion, utan omsampling."""
    outputs = []
    arcpy.SetProgressor("step", "Skriver rasterdata...", 0, len(sizes), 1)
    try:
        for index, tile in enumerate(sorted(sizes)):
            lon, lat = tile
            arcpy.SetProgressorPosition(index)
            arcpy.SetProgressorLabel("Skriver {} ({}/{})".format(
                _grid_name(lon, lat), index + 1, len(sizes)))

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

            emb, scales, landmask = _fetch_tile_files(client, year, lon, lat)
            _tile_raster(emb, scales, landmask, band_indices, out_path)
            outputs.append(out_path)
        arcpy.SetProgressorPosition(len(sizes))
    finally:
        arcpy.ResetProgressor()
    return outputs


def _build_mosaic(client, sizes, year, band_indices, out_gdb,
                  out_name, target_sr, target_ext, overwrite, messages, clip_geom=None):
    """
    En sammanfogad raster i target_sr, klippt till områdets rektangel och, om
    clip_geom (polygon i target_sr) är given, till polygonerna med NoData utanför.

    Varje tile skrivs först i sin egen UTM-projektion och projiceras sedan om mot
    ett gemensamt 10 m-rutnät (snapRaster). Utan snappningen får varje tile ett
    eget origo, och MosaicToNewRaster tvingas omsampla en gång till.

    Klippningen görs med Clip och ClippingGeometry, som fungerar med Basic-licens
    (ExtractByMask kräver Spatial Analyst). Mosaiken skrivs då först under ett
    tillfälligt namn i geodatabasen och klipps till det slutliga.
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

        arcpy.SetProgressor("step", "Bygger raster...", 0, len(sizes), 1)
        for index, tile in enumerate(sorted(sizes)):
            lon, lat = tile
            grid = _grid_name(lon, lat)
            arcpy.SetProgressorPosition(index)
            arcpy.SetProgressorLabel("Bygger {} ({}/{})".format(
                grid, index + 1, len(sizes)))

            emb, scales, landmask = _fetch_tile_files(client, year, lon, lat)
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

        arcpy.SetProgressorPosition(len(sizes))
        arcpy.ResetProgressor()

        if not projected:
            return []

        step = "Sammanfogar {} tiles till {}".format(len(projected), name)
        if clip_geom is not None:
            step += " (steg 1 av 2)"
        messages.addMessage(step + "...")
        arcpy.SetProgressorLabel(step + "...")
        mosaic_name = name
        if clip_geom is not None:
            mosaic_name = arcpy.ValidateTableName(
                "{}_oklippt_{}".format(name, uuid.uuid4().hex[:8]), out_gdb)
            temp_mosaic = os.path.join(out_gdb, mosaic_name)
        # FIRST i stället för BLEND: i överlappen mellan tiles ska ett helt
        # embedding-värde behållas, inte medelvärdet av två.
        arcpy.management.MosaicToNewRaster(
            projected, out_gdb, mosaic_name, target_sr, "32_BIT_FLOAT",
            CELL_SIZE_M, len(band_indices), "FIRST", "FIRST"
        )

        if clip_geom is not None:
            step = "Klipper mosaiken till polygonerna (steg 2 av 2)"
            messages.addMessage(step + "...")
            arcpy.SetProgressorLabel(step + "...")
            try:
                # NoData-värdet lämnas tomt så att mosaikens eget behålls; 0 är ett
                # giltigt embedding-värde och duger inte som NoData.
                arcpy.management.Clip(
                    temp_mosaic,
                    "{} {} {} {}".format(target_ext.XMin, target_ext.YMin,
                                         target_ext.XMax, target_ext.YMax),
                    out_path, clip_geom, "", "ClippingGeometry", "MAINTAIN_EXTENT",
                )
            finally:
                try:
                    arcpy.management.Delete(temp_mosaic)
                except Exception:                           # noqa: BLE001
                    messages.addWarningMessage(
                        "Kunde inte ta bort den tillfälliga rastern {}.".format(temp_mosaic))
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


def _clear_cache(client, sizes, year, messages):
    """Ta bort de hämtade filerna för körningens tiles ur cache-mappen."""
    for tile in sizes:
        lon, lat = tile
        for path in _cached_paths(client, year, lon, lat):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                messages.addWarningMessage(
                    "  Kunde inte ta bort {}: {}".format(path, exc))
