# Tessera embeddings to GDB

ArcGIS Pro Python toolbox that downloads Tessera satellite embeddings for a bounding box into a
file geodatabase.

Tessera is an Earth observation foundation model from the University of Cambridge. It publishes
global annual embeddings built from Sentinel-1 and Sentinel-2: 128 channels per pixel at 10 m
resolution. The data is served as plain HTTPS files from Source Cooperative and needs no API key
or account.

The data is large. One tile covers about 11 km and is roughly 90 MB to download, or 360 MB in
the geodatabase once written as 32-bit float with all 128 bands. The tool therefore shows the
size before anything is fetched, both in the dialog and in the first lines of the run log, and
refuses to start above a configurable limit.

## Requirements

- ArcGIS Pro 3.x. Developed and tested on 3.6 with Python 3.13.
- No dependencies beyond `arcpy`, `numpy` and the standard library.
- Internet access to `data.source.coop`.

## Install

1. Clone or download this repo.
2. In ArcGIS Pro: Catalog, Toolboxes, Add Toolbox, select `TesseraToGDB.pyt`.
3. Open Tessera, Hämta Tessera-raster till geodatabas.

## The tool dialog

The UI is in Swedish, matching a Swedish ArcGIS Pro install.

| Parameter | Default | Notes |
|---|---|---|
| Bounding box | current map view | Area to download |
| Koordinatsystem för bounding boxen | SWEREF99 TM | Used when the extent carries no CRS of its own |
| År | 2024 | 2017 to 2025 for the v1 dataset |
| Dataset-version | v1 | v1 is global. v2 is beta with partial year coverage, v1.1 is Cambridge only |
| Band att spara | all 128 | Accepts `1-16,64`. Reduces geodatabase size, not download size |
| Uppskattad storlek | read only | Download size, geodatabase size and temporary disk use |
| Kontrollera exakt storlek mot servern | off | Queries the server for the true size of every tile |
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

## Notes on the data

- Tiles are a 0.1 degree grid with centres at `k * 0.1 + 0.05`.
- Each tile sits in its own UTM zone. A bounding box in Sweden commonly spans two zones.
- The `.npy` files carry no georeferencing. Coordinate system and origin come from the tile's
  landmask GeoTIFF, which the tool downloads alongside the data.
- Tiles over open water are not published and return HTTP 404. The tool reports how many of the
  requested tiles actually exist and skips the rest.
- Selecting fewer bands does not reduce the download. The whole 128 channel file has to be
  fetched either way.

## Source

- Data: https://data.source.coop/tessera/tessera
- Project: https://geotessera.org/
