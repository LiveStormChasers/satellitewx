"""
STAR GOES full disk -> web-mercator tiles -> one packed object per frame.

The projection is the GOES-R ABI fixed grid (PUG vol 3, 5.1.2.8). Verified
against coastlines before this was written.

Output per frame is two objects, not 27,000:
    <sat>/<frame>.pack   concatenated JPEG tiles
    <sat>/<frame>.idx    binary index: z, x, y -> offset, length
That is what keeps R2 inside the free tier: R2 bills per object write.
"""

import io
import json
import math
import struct
import time
import urllib.request

import numpy as np
from PIL import Image

# --- fixed grid constants -------------------------------------------------
REQ = 6378137.0
RPOL = 6356752.31414
H = 42164160.0
E2 = (REQ * REQ - RPOL * RPOL) / (REQ * REQ)
REQ2_RPOL2 = (REQ * REQ) / (RPOL * RPOL)
RPOL2_REQ2 = (RPOL * RPOL) / (REQ * REQ)
FD_HALF = 0.151872

SATS = {
    "GOES18": {"lon0": -137.0, "star": "GOES18"},   # West
    "GOES19": {"lon0": -75.2, "star": "GOES19"},    # East
}

TILE = 256
IDX_MAGIC = b"LSCT"
IDX_REC = "<BIIQI"          # z, x, y, offset, length
IDX_REC_SIZE = struct.calcsize(IDX_REC)


# --- projection -----------------------------------------------------------
def geodetic_to_scan(lat, lon, lon0):
    """lat/lon (degrees, arrays or scalars) -> (x, y) scan radians, valid mask."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    d = np.radians(((lon - lon0 + 180.0) % 360.0) - 180.0)
    phi = np.radians(lat)

    phi_c = np.arctan(RPOL2_REQ2 * np.tan(phi))
    cosc = np.cos(phi_c)
    sinc = np.sin(phi_c)
    rc = RPOL / np.sqrt(1.0 - E2 * cosc * cosc)

    sx = H - rc * cosc * np.cos(d)
    sy = -rc * cosc * np.sin(d)
    sz = rc * sinc

    visible = H * (H - sx) > (sy * sy + REQ2_RPOL2 * sz * sz)

    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.arctan(sz / sx)
        x = np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz))
    return x, y, visible


def scan_to_pixel(x, y, size):
    """Scan radians -> source pixel coords, pixel-centre convention."""
    s = 2.0 * FD_HALF
    col = (x + FD_HALF) / s * size - 0.5
    row = (FD_HALF - y) / s * size - 0.5
    return col, row


# --- mercator -------------------------------------------------------------
def tile_lonlat(z, tx, ty):
    """Per-pixel lon (row vector) and lat (column vector) for one tile."""
    n = TILE * (2 ** z)
    px = tx * TILE + np.arange(TILE, dtype=np.float64) + 0.5
    py = ty * TILE + np.arange(TILE, dtype=np.float64) + 0.5
    lon = px / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * py / n))))
    return lon, lat


def tiles_covering_disk(z, lon0):
    """Tiles at zoom z whose corners include at least one visible point."""
    n = 2 ** z
    ys, xs = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    world = TILE * n
    lon = (xs * TILE) / world * 360.0 - 180.0
    with np.errstate(over="ignore"):
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (ys * TILE) / world))))
    _, _, vis = geodetic_to_scan(lat, lon, lon0)
    # a tile is kept if any of its four corners is on the disk
    keep = vis[:-1, :-1] | vis[:-1, 1:] | vis[1:, :-1] | vis[1:, 1:]
    tys, txs = np.nonzero(keep)
    return list(zip(txs.tolist(), tys.tolist()))


# --- rendering ------------------------------------------------------------
def render_tile(src, size, lon0, z, tx, ty):
    """One tile as an RGB array plus a coverage fraction, or None if empty."""
    lon, lat = tile_lonlat(z, tx, ty)
    lon2 = np.broadcast_to(lon[None, :], (TILE, TILE))
    lat2 = np.broadcast_to(lat[:, None], (TILE, TILE))

    x, y, vis = geodetic_to_scan(lat2, lon2, lon0)
    col, row = scan_to_pixel(x, y, size)

    inb = vis & (col >= 0) & (row >= 0) & (col <= size - 1) & (row <= size - 1)
    if not inb.any():
        return None, 0.0

    c = np.where(inb, col, 0.0)
    r = np.where(inb, row, 0.0)
    c0 = np.floor(c).astype(np.int32)
    r0 = np.floor(r).astype(np.int32)
    c1 = np.minimum(c0 + 1, size - 1)
    r1 = np.minimum(r0 + 1, size - 1)
    fc = (c - c0)[..., None]
    fr = (r - r0)[..., None]

    p00 = src[r0, c0].astype(np.float32)
    p10 = src[r0, c1].astype(np.float32)
    p01 = src[r1, c0].astype(np.float32)
    p11 = src[r1, c1].astype(np.float32)

    top = p00 * (1 - fc) + p10 * fc
    bot = p01 * (1 - fc) + p11 * fc
    out = top * (1 - fr) + bot * fr
    out[~inb] = 0

    return out.astype(np.uint8), float(inb.mean())


# --- source ---------------------------------------------------------------
def fetch_disk(sat, size, timeout=180):
    """Download a STAR full disk and return (array, size, url)."""
    star = SATS[sat]["star"]
    base = f"https://cdn.star.nesdis.noaa.gov/{star}/ABI/FD/GEOCOLOR/"
    url = base + (f"latest_{star}-ABI-FD-GEOCOLOR-{size}x{size}.jpg"
                  if size else "latest.jpg")
    req = urllib.request.Request(url, headers={"User-Agent": "LiveStormChasers-satellitewx"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.width != img.height:
        raise SystemExit(f"{url} is {img.width}x{img.height}, not square — "
                         "that file has a banner and is not the bare fixed grid")
    return np.asarray(img), img.width, url


# --- packing --------------------------------------------------------------
class Pack:
    """Concatenated tiles plus a binary index. Two objects per frame."""

    def __init__(self):
        self.buf = io.BytesIO()
        self.records = []

    def add(self, z, x, y, blob):
        off = self.buf.tell()
        self.buf.write(blob)
        self.records.append((z, x, y, off, len(blob)))

    def index_bytes(self):
        self.records.sort()
        out = io.BytesIO()
        out.write(IDX_MAGIC)
        out.write(struct.pack("<I", len(self.records)))
        for rec in self.records:
            out.write(struct.pack(IDX_REC, *rec))
        return out.getvalue()

    def pack_bytes(self):
        return self.buf.getvalue()


def build(sat, size, minz, maxz, quality=82, min_coverage=0.002, progress=True):
    src, size, url = fetch_disk(sat, size)
    lon0 = SATS[sat]["lon0"]
    pack = Pack()
    stats = {}

    for z in range(minz, maxz + 1):
        t0 = time.time()
        kept = 0
        for tx, ty in tiles_covering_disk(z, lon0):
            arr, cov = render_tile(src, size, lon0, z, tx, ty)
            if arr is None or cov < min_coverage:
                continue
            b = io.BytesIO()
            Image.fromarray(arr).save(b, format="JPEG", quality=quality, optimize=True)
            pack.add(z, tx, ty, b.getvalue())
            kept += 1
        stats[z] = {"tiles": kept, "seconds": round(time.time() - t0, 1)}
        if progress:
            print(f"  z{z}: {kept} tiles in {stats[z]['seconds']}s", flush=True)

    return pack, {"satellite": sat, "source": url, "source_size": size,
                  "lon0": lon0, "minzoom": minz, "maxzoom": maxz, "zooms": stats}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--satellite", default="GOES19", choices=sorted(SATS))
    ap.add_argument("--size", type=int, default=10848,
                    help="STAR tier: 1808, 5424, 10848, 21696. 0 = latest.jpg")
    ap.add_argument("--min-zoom", type=int, default=3)
    ap.add_argument("--max-zoom", type=int, default=7)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    pack, meta = build(args.satellite, args.size, args.min_zoom, args.max_zoom)
    stem = os.path.join(args.out, args.satellite.lower())
    open(stem + ".pack", "wb").write(pack.pack_bytes())
    open(stem + ".idx", "wb").write(pack.index_bytes())
    meta["tiles"] = len(pack.records)
    meta["pack_bytes"] = len(pack.pack_bytes())
    open(stem + ".json", "w").write(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
