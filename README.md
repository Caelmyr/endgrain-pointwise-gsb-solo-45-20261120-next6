# Endgrain Lab

Endgrain Lab is a local workbench for inspecting wood end-face photographs. It keeps each original image with an editable measuring setup, revision-bound analyses, observations, and issued reports.

## Start

```sh
docker compose up -d --build --wait
```

Open http://127.0.0.1:18948. Use `ENDGRAIN_PORT` to select another loopback port. Compose stores originals, measuring state, and reports in the `inspections` volume. `docker compose stop` and `docker compose down` leave the volume available for a later start.

For a Python 3.11 environment, install `requirements.txt` and run `./start.sh`. The default local data directory is `.endgrain/`. Set `ENDGRAIN_DATA` to use another directory. The Docker build can take an `ENDGRAIN_DEPS_IMAGE` build argument for a cached Python 3.11 image containing the pinned dependencies. Use the same architecture and Python ABI as the runtime base.

## Typical inspection

1. Import a single face photograph or open the synthetic samples. The synthetic white oak has a branching check and growth-ring bands. The synthetic walnut includes a brighter area and two tones. The synthetic ash has periodic saw marks.
2. Calibrate with a measured ruler or rectify four corners of a known rectangle. Click the source points in order. The Grid view displays the selected plane and the ruler length or grid spacing. Draw a simple polygon using **Region** to limit analysis.
3. Optionally attach a flat-field reference from the same setup. The reference must have the same oriented dimensions. Use the Lighting view to check gain, clipping and which locations were measured on the board, estimated from nearby board pixels, or kept on the original capture.
4. Run a measurement. Results include an overlay, summary, and a downloadable CSV. Use the Pixel probe on the working image for a magnified patch, source mapping and luminance profiles. Add review markers by clicking locations in the selected measuring region.
5. Compare two saved measurements of the same kind. A color difference overlay is available when they share a source, plane, region and lighting settings. The Recipes tab can replay selected tool settings on a second inspection. The Lots tab screens saved readings and exports a slide of face overlays and a CSV.
6. Include current measurements in a report and download its evidence ZIP. It contains source material, the measuring setup illustration, overlays, tables, the manifest and checksums. The separate inspection exchange ZIP carries the original capture, settings, attached lighting reference and current markers into a separate record on restore.

## Image measurements

- **Capture quality** shows channel clipping, low-detail regions, glare-like pixels and supported coverage.
- **Annual ring transect** samples a narrow band and suggests contrast boundaries. Inspect or replace the candidates before interpreting the distances as growth increments.
- **Crack traces** estimates skeleton lengths and dark-region widths. It identifies branch and edge-touching regions as review candidates.
- **Vessel pores** estimates component diameter, area and density from compact dark regions.
- **Grain orientation** maps axial texture and represents ambiguous local orientation separately from reliable regions.
- **Saw chatter** measures texture power within an axis and wavelength band in overlapping windows.
- **Wood footprint** isolates a warm, bright connected end-face silhouette against its backing.
- **Edge recession** shows supported loss or unknown areas along a manually specified nominal stock outline.
- **Color zones** compares appearance to swatches in display-referred Lab using the current correction settings.

## Lighting reference handling

A flat-field reference is rarely perfect, so each reference pixel is placed in one of three classes, shown separately in the Lighting view and reported on the frame statistics:

- **Board measured** — reference pixels inside the trusted tonal range whose smoothing window is well supported. These drive the gain field exactly as the original flat-field method.
- **Estimated from nearby board** — glare, transparent dead spots and alpha holes that still have enough measured board around them. A two-scale normalized-convolution fill is blended by wide-window evidence and distance to the nearest board pixel, so a compact chip or reflection transitions smoothly instead of leaving a dark seam.
- **Original kept** — locations too far from any measured board pixel, such as the centre of a large damaged area. The uncorrected sample is retained there and feathered toward the corrected rim, rather than guessing a field with no local evidence.

Highlight protection (the linear-light *knee*) rolls off the brightening part of the gain on already-bright sample pixels, so recoverable highlights and fine dark lines are not pushed white; darkening corrections and mid-tone wood are unaffected. Set the knee to 1 to apply gain without protection. The same correction object feeds the working image, every measurement, the Lighting/working downloads and the stored run snapshots, so changing the calibration or region never recomputes a different field. With a complete, clean reference the measured path is byte-identical to the original correction.

## Acquisition limits

Inputs are PNG, JPEG, TIFF and WebP, one frame per import, up to 32 MiB and 16 megapixels with each side at least 48 pixels. The measuring raster has a maximum side of 1,800 pixels. Reported pixel widths use that raster. Fully opaque source pixels contribute to geometric support. EXIF orientation is applied before choosing coordinates and high bit-depth grayscale inputs are rendered to an 8-bit measuring representation. The unmodified input remains downloadable.

The rectangle method assumes a planar specimen and a known rectangular reference in that plane. Neither scale method corrects lens distortion. Inspection tools are approximate image measurements for operator review. The demonstration stock is procedurally generated and cannot establish botanical identification, defect acceptance limits or a calibrated color standard.

A change to measuring setup starts a new revision. Older results stay in measurement history, and issued reports remain bound to the original revision and settings. A measurement recipe transfers settings, not the identity of a manual ring or defect.

## Source layout

`endgrain/` contains image ingest, geometry, measurement, SQLite, and report logic. `static/` contains the browser workbench. `app.py` runs the Waitress server. `scripts/count_backend.py` can count Python backend source.
