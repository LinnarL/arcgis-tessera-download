# Tessera embeddings to GDB

ArcGIS Pro Python toolbox that downloads Tessera satellite embeddings for a bounding box into a
file geodatabase.

Tessera is an Earth observation foundation model from the University of Cambridge. It publishes
global annual embeddings built from Sentinel-1 and Sentinel-2: 128 channels per pixel at 10 m
resolution. Downloads and the tile registry are handled by the `geotessera` package, which reads
from the project's public S3 bucket. No API key or account is needed.

The data is large. One tile covers about 11 km and is roughly 90 MB to download, or 360 MB in
the geodatabase once written as 32-bit float with all 128 bands. The tool therefore shows the
size before anything is fetched, both in the dialog and in the first lines of the run log, and
refuses to start above a configurable limit.

## Requirements

- ArcGIS Pro 3.x. Developed and tested on 3.6 with Python 3.13.
- The `geotessera` package, which handles the tile registry and downloads.
- Internet access to `s3.us-west-2.amazonaws.com`.

`geotessera` is not part of the default `arcgispro-py3` environment, and ArcGIS Pro
does not allow installing into it. Clone the environment first.

## Install

1. In ArcGIS Pro: Settings, Package Manager, clone the active environment. Name the
   clone something like `arcgispro-py3-personal` and make it active.
2. Install geotessera into the clone, then pin pyarrow back to the version Pro
   supports. geotessera pulls a newer pyarrow that fails to load once `arcpy` is
   imported, which breaks the registry:

   ```
   python -m pip install geotessera
   python -m pip install "pyarrow==20.0.0"
   ```

3. Clone or download this repo.
4. In ArcGIS Pro: Catalog, Toolboxes, Add Toolbox, select `TesseraToGDB.pyt`.
5. Open Tessera, Hämta Tessera-raster till geodatabas.

If the environment is wrong the tool stops with a message naming the active
environment rather than failing part way through a download.

## The tool dialog

The UI is in Swedish, matching a Swedish ArcGIS Pro install.

| Parameter | Default | Notes |
|---|---|---|
| Bounding box | current map view | Area to download |
| Koordinatsystem för bounding boxen | the map's CRS | How the extent numbers are read. ArcGIS hands the tool an extent without a CRS, so this is the tool's only way to know |
| År | 2024 | 2017 to 2025 for the v1 dataset |
| Dataset-version | v1 | v1 is global. v2 is beta with partial year coverage, v1.1 is Cambridge only |
| Band att spara | all 128 | Accepts `1-16,64`. Reduces geodatabase size, not download size |
| Uppskattad storlek | read only | Download size, geodatabase size and temporary disk use |
| Kontrollera exakt storlek mot servern | off | Uses the registry to report only the tiles that really exist and their true sizes |
| Utdata-geodatabas | project default | Must be a file geodatabase |
| Namn på utdata-raster | `tessera_<år>` | Follows the year until you type your own name |
| Utdataform | mosaik | Single merged raster, or one raster per tile in native UTM |
| Koordinatsystem för mosaiken | SWEREF99 TM | Mosaic mode only |
| Skriv över befintlig raster | on | |
| Avbryt om nedladdningen överstiger | 5 GB | Hard stop before anything is downloaded |
| Cache-mapp för nedladdade tiles | system temp | Reused across runs. Avoid cloud-synced folders |
| Behåll nedladdade tiles | on | |
| Lägg till resultatet i kartan | on | |

## Output

Mosaic mode reprojects every tile to the chosen coordinate system on a shared 10 m grid and
merges them into one raster clipped to the bounding box. Overlaps keep the first tile's values
rather than blending, so no pixel holds an averaged embedding vector.

Tile mode writes one raster per tile in the tile's own UTM zone with no resampling. Use it when
you want the values untouched.

Values are 32-bit float. Tessera stores embeddings quantised as int8 with one scale factor per
pixel, and the tool multiplies them out before writing.

## Coordinate systems

ArcGIS passes the bounding box to the tool as four numbers with no coordinate system attached,
so the tool cannot detect which CRS you drew the box in. It assumes the active map's CRS, which
is right in almost every case, and prints the assumption on the first line of the run log:

```
Bounding box tolkas som SWEREF99_18_00 (EPSG:3011): 173565.35, 6578601.74 till 177565.35, 6582601.74
```

If that line names the wrong system, set Koordinatsystem för bounding boxen yourself. Getting it
wrong does not fail, it downloads a different part of the world.

The output CRS is separate and defaults to SWEREF99 TM in mosaic mode. Set Koordinatsystem för
mosaiken if you want the raster in your project's own CRS instead.

## Downloads and caching

Downloads go through geotessera, which reads from the project's S3 bucket. Tiles are cached in
the folder given by Cache-mapp för nedladdade tiles and reused across runs, so a repeated or
resumed run only fetches what is missing.

The first run also downloads a manifest listing every published tile. It takes a while and is
cached afterwards. That manifest is what makes the size estimate exact without any extra
requests: the tool asks the registry which tiles exist and how large they are before fetching
anything.

## Notes on the data

- Tiles are a 0.1 degree grid with centres at `k * 0.1 + 0.05`.
- Each tile sits in its own UTM zone. A bounding box in Sweden commonly spans two zones.
- The `.npy` files carry no georeferencing. Coordinate system and origin come from the tile's
  landmask GeoTIFF, which the tool downloads alongside the data.
- Tiles over open water are not published. The registry knows which tiles exist, so the tool
  reports how many of the requested tiles are available and skips the rest before downloading.
- Selecting fewer bands does not reduce the download. The whole 128 channel file has to be
  fetched either way.

## Source

- Project: https://geotessera.org/
- Library: https://github.com/ucam-eo/geotessera
