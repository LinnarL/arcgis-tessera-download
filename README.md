# Tessera hämta embeddings

ArcGIS Pro Python toolbox that downloads Tessera satellite embeddings for an area into a file
geodatabase. The area is either polygons (from a layer, or drawn in the map) or an extent (the map
view, a layer's extent, a drawn rectangle or typed coordinates).

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
4. In ArcGIS Pro: Catalog, Toolboxes, Add Toolbox, select `TesseraHamtaEmbeddings.pyt`.
5. Open Tessera, Hämta Tessera-raster till geodatabas.

If the environment is wrong the tool stops with a message naming the active
environment rather than failing part way through a download.

## The tool dialog

The UI is in Swedish, matching a Swedish ArcGIS Pro install.

| Parameter | Default | Notes |
|---|---|---|
| Avgränsa området med | Utbredning | `Polygoner (lager eller ritade i kartan)` or `Utbredning (kartvy, lager eller koordinater)`. Only the chosen input below is enabled |
| Intresseområde (polygoner) | empty | Polygon layer (a selection is honoured) or polygons drawn in the dialog. Any CRS. Polygon mode only |
| Utbredning | current map view | Rectangle to download. Extent mode only |
| Koordinatsystem för utbredningen | the map's CRS | How typed extent numbers are read when the extent carries no CRS of its own. Extent mode only |
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
| Cache-mapp för nedladdade tiles | `%LOCALAPPDATA%\Tessera_nedladdning` | Where tiles are downloaded to. Avoid cloud-synced folders |
| Behåll nedladdade tiles | off | On keeps the tiles so a re-run skips the download, at hundreds of MB per tile |
| Lägg till resultatet i kartan | on | |

## Output

Mosaic mode reprojects every tile to the chosen coordinate system on a shared 10 m grid and
merges them into one raster clipped to the area's rectangle. Overlaps keep the first tile's values
rather than blending, so no pixel holds an averaged embedding vector.

With polygons, only tiles that intersect the polygons themselves are downloaded, not every tile
in their bounding box. The mosaic covers the polygons' bounding box and every cell outside the
polygons is NoData. The clip uses the Clip tool with clipping geometry, which needs no Spatial
Analyst licence. It writes the unclipped mosaic under a temporary name in the geodatabase first,
so the geodatabase briefly needs twice the final size.

Tile mode writes one raster per tile in the tile's own UTM zone with no resampling. Tiles are
written whole: they are not clipped to the extent or to the polygons. Use it when you want the
values untouched.

Values are 32-bit float. Tessera stores embeddings quantised as int8 with one scale factor per
pixel, and the tool multiplies them out before writing.

## Coordinate systems

Polygons always carry their own coordinate system and need no setting.

An extent picked from a layer or dataset carries its CRS. Typed coordinates and the map view's
extent reach the tool as four bare numbers, so the tool has to assume a CRS: Koordinatsystem för
utbredningen, which defaults to the active map's CRS (SWEREF 99 TM when there is no map). The run
log names the CRS used and where it came from:

```
Utbredningen tolkas som SWEREF99_18_00 (EPSG:3011) (den aktiva kartans koordinatsystem): 173565.35, 6578601.74 till 177565.35, 6582601.74
```

If that line names the wrong system, set Koordinatsystem för utbredningen yourself. Getting it
wrong does not fail, it downloads a different part of the world.

The output CRS is separate and defaults to SWEREF99 TM in mosaic mode. Set Koordinatsystem för
mosaiken if you want the raster in your project's own CRS instead.

## Scripting

Parameter names: `aoi_mode`, `aoi`, `extent`, `extent_crs`, `year`, `dataset`, `bands`,
`estimate`, `exact_estimate`, `out_gdb`, `out_name`, `out_mode`, `target_crs`, `overwrite`,
`max_gb`, `cache_dir`, `keep_cache`, `add_to_map`. The area parameters were added in front of the
old ones, so call the tool with keyword arguments. The older label `Polygoner i ett lager` is still
accepted for `aoi_mode`.

```python
arcpy.ImportToolbox(r"...\TesseraHamtaEmbeddings.pyt")
arcpy.tessera.HamtaTesseraRaster(aoi_mode="Polygoner (lager eller ritade i kartan)",
                                 aoi="my_polygon_layer", out_gdb=r"C:\data\out.gdb",
                                 out_name="tessera_2024")
```

## Downloads and caching

Downloads go through geotessera, which reads from the project's S3 bucket. By default the
downloaded tiles are deleted once the raster is written, so nothing accumulates. Tick Behåll
nedladdade tiles to keep them, which makes a re-run over the same area skip the download at the
cost of a few hundred MB per tile.

The download folder deliberately sits under `%LOCALAPPDATA%` rather than the system temp
directory. Inside ArcGIS Pro the temp directory is a per-session folder that changes on every
restart and is often left behind, so a cache there could never be reused.

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
