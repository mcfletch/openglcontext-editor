# Elevation data: the SRTM height file, and putting it on the ground

Facts this package's DEM import relies on, and where each came from. Nothing
here was taken from a GIS tool's source; the layout facts are from the published
mission documentation for the data, and the geodesy is textbook.

## Sources

- **S1.** NASA/USGS Shuttle Radar Topography Mission (SRTM) data products and
  their user documentation, published by the mission and its distributors. The
  `.hgt` height file's layout, sample order, resolutions, void value and file
  naming are all stated there.
- **S2.** WGS 84, as defined by NGA TR8350.2 / EPSG:4326 — the ellipsoid the
  coordinates are given on, and its semi-major axis and flattening.
- **S3.** Standard geodesy for the radii of curvature of an ellipsoid, and for
  a local tangent-plane (east/north) approximation about a point. This is
  textbook mathematics, not anyone's code.

## 1. The `.hgt` height file (S1)

1.1 A `.hgt` file is **raw samples with no header**: signed 16-bit integers,
    **big-endian**, in metres above the WGS 84 ellipsoid-referenced vertical
    datum the product is published against.

1.2 The samples form a **square grid of N x N**, where N follows from the file's
    size: `N = sqrt(bytes / 2)`. The published resolutions are **1201** (three
    arc-seconds a sample) and **3601** (one arc-second a sample).

1.3 The grid is stored **row by row from north to south**, and within a row
    **west to east**. So the first sample in the file is the **north-west**
    corner and the last is the **south-east** corner.

1.4 The grid **covers one degree of latitude by one degree of longitude**, and
    the first and last rows and columns lie **on** the tile's edges. Adjacent
    tiles therefore repeat one row/column of samples along their shared edge:
    the sample spacing is `1 / (N - 1)` degrees, not `1 / N`.

1.5 A sample with no data is **-32768**.

1.6 The file is named for the **south-west corner** of the degree square it
    covers, as a latitude with a hemisphere letter and a longitude with one:
    `N47E008.hgt` is the square whose south-west corner is 47 degrees north,
    8 degrees east. Latitude is two digits, longitude three.

## 2. Geographic coordinates on the ground (S2, S3)

2.1 Coordinates are latitude and longitude on the **WGS 84 ellipsoid**, whose
    semi-major axis is **a = 6378137.0 m** and whose flattening is
    **f = 1 / 298.257223563**. The first eccentricity squared is
    `e2 = f * (2 - f)`.

2.2 At a latitude `phi`, the **meridional** radius of curvature (north-south)
    and the **prime-vertical** radius (east-west) are

        M(phi) = a * (1 - e2) / (1 - e2 * sin^2(phi))^(3/2)
        N(phi) = a / sqrt(1 - e2 * sin^2(phi))

2.3 About a centre `(phi0, lambda0)`, a **local tangent plane** in metres is

        east  = (lambda - lambda0) * N(phi0) * cos(phi0)
        north = (phi    - phi0)    * M(phi0)

    with the angles in radians. This is a first-order approximation valid over
    a region small enough that the radii do not change appreciably across it --
    a few tens of kilometres, which is the size of anything a track editor
    frames. It is not a projection to use across a continent.

## 3. What this package does with it

3.1 A world's axes are metres, with **+x east**, **+y up** and **-z north**, so
    the north of 2.3 is `-z`.

3.2 Elevations are metres already, so the only vertical decision is the datum
    the world's zero sits at and any deliberate scaling of the relief.
