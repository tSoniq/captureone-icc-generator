#!/usr/bin/env python3
"""
bake_icc_hald.py — Combine an ICC camera input profile with a Hald CLUT
into a single ICC input profile.

Usage:
    python3 merge-camera-icc-and-hald.py input_camera.icc hald_clut.png -o combined.icc [-s <n>]

Requirements:
    pip install pillow numpy

How it works:
    1. Generates a dense 3D grid of RGB device values (simulating camera output).
    2. Transforms each value through the existing ICC camera profile → sRGB.
    3. Applies the Hald CLUT (trilinear interpolation) to the sRGB values.
    4. Builds a new ICC profile whose A2B (device→PCS) table maps the original
       camera RGB directly to the CLUT-modified Lab values.

The resulting profile can replace the original camera input profile in any
ICC-aware raw processor, and it will include the look from the Hald CLUT.
"""

import argparse
import struct
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageCms import (
    ImageCmsProfile,
    buildTransform,
    createProfile,
    getProfileDescription,
    getProfileInfo,
)


# ---------------------------------------------------------------------------
# Hald CLUT loading and application
# ---------------------------------------------------------------------------

def load_hald_clut(path: str) -> tuple[np.ndarray, int]:
    """Load a Hald CLUT PNG and return (clut_data, level).

    A Hald CLUT of level L is an image of (L*L) x (L*L) pixels encoding
    a 3D LUT with L*L nodes per axis (total L^6 entries, i.e. L^3 x L^3
    pixels).  Level is derived from the image width.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w != h:
        raise ValueError(f"Hald CLUT must be square, got {w}x{h}")

    # Determine level: w = level^3, so level = round(w^(1/3))
    level = round(w ** (1.0 / 3.0))
    expected = level ** 3
    if expected != w:
        raise ValueError(
            f"Image size {w}x{h} does not match any Hald CLUT level "
            f"(closest level={level} expects {expected}x{expected})"
        )

    clut = np.asarray(img, dtype=np.float64) / 255.0
    clut = clut.reshape(-1, 3)  # flatten to (N, 3)
    return clut, level


def apply_hald_clut(rgb: np.ndarray, clut: np.ndarray, level: int) -> np.ndarray:
    """Apply a Hald CLUT via trilinear interpolation.

    Parameters
    ----------
    rgb : (N, 3) float64 array, values in [0, 1], assumed sRGB.
    clut : flattened CLUT data from load_hald_clut.
    level : Hald CLUT level.

    Returns
    -------
    (N, 3) float64 array of transformed sRGB values in [0, 1].
    """
    size = level * level  # nodes per axis
    rgb_scaled = np.clip(rgb, 0.0, 1.0) * (size - 1)

    r0 = np.floor(rgb_scaled[:, 0]).astype(np.intp)
    g0 = np.floor(rgb_scaled[:, 1]).astype(np.intp)
    b0 = np.floor(rgb_scaled[:, 2]).astype(np.intp)
    r1 = np.minimum(r0 + 1, size - 1)
    g1 = np.minimum(g0 + 1, size - 1)
    b1 = np.minimum(b0 + 1, size - 1)

    rf = (rgb_scaled[:, 0] - r0)[:, np.newaxis]
    gf = (rgb_scaled[:, 1] - g0)[:, np.newaxis]
    bf = (rgb_scaled[:, 2] - b0)[:, np.newaxis]

    def idx(r, g, b):
        # Hald CLUT pixel order: R changes fastest, then G, then B
        return r + g * size + b * size * size

    c000 = clut[idx(r0, g0, b0)]
    c100 = clut[idx(r1, g0, b0)]
    c010 = clut[idx(r0, g1, b0)]
    c110 = clut[idx(r1, g1, b0)]
    c001 = clut[idx(r0, g0, b1)]
    c101 = clut[idx(r1, g0, b1)]
    c011 = clut[idx(r0, g1, b1)]
    c111 = clut[idx(r1, g1, b1)]

    result = (
        c000 * (1 - rf) * (1 - gf) * (1 - bf)
        + c100 * rf * (1 - gf) * (1 - bf)
        + c010 * (1 - rf) * gf * (1 - bf)
        + c110 * rf * gf * (1 - bf)
        + c001 * (1 - rf) * (1 - gf) * bf
        + c101 * rf * (1 - gf) * bf
        + c011 * (1 - rf) * gf * bf
        + c111 * rf * gf * bf
    )
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ICC profile construction helpers (raw tag writing via struct)
# ---------------------------------------------------------------------------
# Pillow's ImageCms can *apply* profiles but can't create arbitrary LUT-based
# profiles.  We build the ICC binary directly — specifically an N-component
# LUT-based input profile using an mAB tag for the A2B0 table.
#
# The approach: we build a 3D CLUT that maps camera RGB → Lab, encoding the
# combined transform (original ICC + Hald CLUT) into a single A2B0 tag.
# ---------------------------------------------------------------------------

# ICC spec constants
ICC_MAGIC = b"acsp"
ICC_PROFILE_VERSION = 0x02400000  # v2.4.0 — widest compatibility
ICC_INPUT_DEVICE = b"scnr"       # scanner/input device class
ICC_RGB_SPACE = b"RGB "
ICC_LAB_SPACE = b"Lab "
ICC_PCS_XYZ = b"XYZ "


def _pad4(data: bytes) -> bytes:
    """Pad data to 4-byte boundary."""
    r = len(data) % 4
    return data + b"\x00" * ((4 - r) % 4)


def _make_xyz_tag(x: float, y: float, z: float) -> bytes:
    """Build an XYZ tag (type 'XYZ ', one triplet)."""
    # s15Fixed16Number: value * 65536
    def s15f16(v):
        return struct.pack(">i", int(round(v * 65536)))

    return b"XYZ " + b"\x00" * 4 + s15f16(x) + s15f16(y) + s15f16(z)


def _make_xyz_tag_exact(x_fixed: int, y_fixed: int, z_fixed: int) -> bytes:
    """Build an XYZ tag from pre-computed s15Fixed16Number values."""
    return b"XYZ " + b"\x00" * 4 + struct.pack(">iii", x_fixed, y_fixed, z_fixed)


def _make_desc_tag(text: str) -> bytes:
    """Build a textDescriptionType tag (ICC v2) for 'desc'."""
    encoded = text.encode("ascii", errors="replace") + b"\x00"
    length = len(encoded)
    # desc tag: sig(4) + reserved(4) + ascii_count(4) + ascii + unicode_code(4)
    # + unicode_count(4) + scriptcode_code(2) + scriptcode_count(1) + data(67)
    tag = b"desc" + b"\x00" * 4
    tag += struct.pack(">I", length)
    tag += encoded
    # Unicode & ScriptCode (empty)
    tag += b"\x00" * 4  # unicode language code
    tag += struct.pack(">I", 0)  # unicode count
    tag += b"\x00" * 2  # scriptcode code
    tag += struct.pack(">B", 0)  # scriptcode count
    tag += b"\x00" * 67  # scriptcode data
    return tag


def _make_text_tag(text: str) -> bytes:
    """Build a textType tag (ICC v2) for 'cprt' and similar.

    ICC signature 'text', layout: sig(4) + reserved(4) + ascii data (null-terminated).
    """
    encoded = text.encode("ascii", errors="replace") + b"\x00"
    return b"text" + b"\x00" * 4 + encoded


def _make_clut_a2b0(clut_rgb_to_lab: np.ndarray, grid_points: int) -> bytes:
    """Build an mft2 (lut16Type) A2B0 tag.

    This is the ICC v2 way of encoding a multidimensional LUT.

    Parameters
    ----------
    clut_rgb_to_lab : (grid_points^3, 3) float64
        Lab values (L 0–100, a/b ±128) for each grid node, ordered with
        the first input channel changing fastest.
    grid_points : int
        Number of grid nodes per input channel.
    """
    n_input = 3
    n_output = 3
    n_entries_in = 256    # input table entries per channel
    n_entries_out = 256   # output table entries per channel

    # --- Header ---
    sig = b"mft2"  # lut16Type signature
    reserved = b"\x00" * 4
    # lut16Type layout after sig+reserved:
    #   inputChan(1) + outputChan(1) + clutPoints(1) + padding(1)
    header = struct.pack(
        ">BBBB",
        n_input,       # input channels
        n_output,      # output channels
        grid_points,   # CLUT grid points
        0,             # padding byte
    )
    # Identity 3x3 matrix (s15Fixed16Number, row-major).
    # For A2B with Lab PCS the matrix must be pure identity.
    # Use exact fixed-point values: 1.0 = 0x00010000, 0.0 = 0x00000000
    matrix = b""
    for i in range(3):
        for j in range(3):
            matrix += struct.pack(">i", 0x00010000 if i == j else 0x00000000)

    # Input / output table sizes
    table_sizes = struct.pack(">HH", n_entries_in, n_entries_out)

    # --- Input curves (identity) ---
    input_tables = b""
    for _ch in range(n_input):
        for i in range(n_entries_in):
            input_tables += struct.pack(">H", int(round(i / (n_entries_in - 1) * 65535)))

    # --- CLUT values ---
    # Encode Lab to ICC 16-bit encoding:
    #   L: 0–100 → 0–65535  (L_encoded = L / 100 * 65535)
    #   a: -128–127 → 0–65535 (a_encoded = (a + 128) / 255 * 65535)
    #   b: same as a
    clut_data = b""
    L = clut_rgb_to_lab[:, 0]
    a = clut_rgb_to_lab[:, 1]
    b = clut_rgb_to_lab[:, 2]
    L_enc = np.clip(L / 100.0 * 65535.0, 0, 65535).astype(np.uint16)
    a_enc = np.clip((a + 128.0) / 255.0 * 65535.0, 0, 65535).astype(np.uint16)
    b_enc = np.clip((b + 128.0) / 255.0 * 65535.0, 0, 65535).astype(np.uint16)
    for i in range(len(L_enc)):
        clut_data += struct.pack(">HHH", L_enc[i], a_enc[i], b_enc[i])

    # --- Output curves (identity) ---
    output_tables = b""
    for _ch in range(n_output):
        for i in range(n_entries_out):
            output_tables += struct.pack(">H", int(round(i / (n_entries_out - 1) * 65535)))

    tag = sig + reserved + header + matrix + table_sizes + input_tables + clut_data + output_tables
    return tag


def build_icc_profile(
    clut_lab: np.ndarray,
    grid_points: int,
    description: str = "Camera + Hald CLUT",
    copyright_text: str = "No copyright",
) -> bytes:
    """Build a complete ICC v2.4 input (scanner) profile with the given A2B CLUT.

    Parameters
    ----------
    clut_lab : (grid_points^3, 3) Lab values for the A2B0 CLUT.
    grid_points : number of grid steps per channel.
    description : profile description string.
    copyright_text : copyright tag string.

    Returns
    -------
    bytes : the complete ICC profile.
    """
    # Prepare tags
    tags = {}
    tags[b"desc"] = _pad4(_make_desc_tag(description))
    tags[b"cprt"] = _pad4(_make_text_tag(copyright_text))

    # D50 media white point — exact ICC spec s15Fixed16 values
    # X=0xF6D6/0x10000, Y=0x10000/0x10000, Z=0xD32D/0x10000
    tags[b"wtpt"] = _pad4(_make_xyz_tag_exact(0x0000F6D6, 0x00010000, 0x0000D32D))

    # A2B0 (the main transform)
    tags[b"A2B0"] = _pad4(_make_clut_a2b0(clut_lab, grid_points))

    # --- Assemble the profile ---
    # Header: 128 bytes
    # Tag table: 4 + 12 * n_tags bytes
    # Then tag data

    n_tags = len(tags)
    tag_table_offset = 128
    tag_table_size = 4 + 12 * n_tags
    data_offset = tag_table_offset + tag_table_size
    # Align data_offset to 4 bytes
    data_offset += (4 - data_offset % 4) % 4

    # Layout tag data
    tag_entries = []
    tag_data = b""
    for sig, data in tags.items():
        offset = data_offset + len(tag_data)
        size = len(data)
        tag_entries.append((sig, offset, size))
        tag_data += data

    profile_size = data_offset + len(tag_data)

    # Build header (128 bytes)
    header = bytearray(128)
    struct.pack_into(">I", header, 0, profile_size)           # profile size
    header[4:8] = b"lcms"                                      # preferred CMM
    struct.pack_into(">I", header, 8, ICC_PROFILE_VERSION)     # version
    header[12:16] = ICC_INPUT_DEVICE                           # device class
    header[16:20] = ICC_RGB_SPACE                              # color space
    header[20:24] = ICC_LAB_SPACE                              # PCS
    # Date/time (now)
    now = time.gmtime()
    struct.pack_into(">HHHHHH", header, 24,
                     now.tm_year, now.tm_mon, now.tm_mday,
                     now.tm_hour, now.tm_min, now.tm_sec)
    header[36:40] = ICC_MAGIC                                  # 'acsp'
    header[40:44] = b"APPL"                                    # primary platform
    struct.pack_into(">I", header, 44, 0)                      # flags
    header[48:52] = b"\x00" * 4                                # device manufacturer
    header[52:56] = b"\x00" * 4                                # device model
    header[56:64] = b"\x00" * 8                                # device attributes
    struct.pack_into(">I", header, 64, 0)                      # rendering intent
    # PCS illuminant (D50) — exact ICC spec s15Fixed16 values
    struct.pack_into(">iii", header, 68, 0x0000F6D6, 0x00010000, 0x0000D32D)
    header[80:84] = b"lcms"                                    # profile creator
    # Bytes 84–128: profile ID + reserved (leave zeroed)

    # Build tag table
    tag_table = struct.pack(">I", n_tags)
    for sig, offset, size in tag_entries:
        tag_table += sig + struct.pack(">II", offset, size)

    # Pad between tag table and data
    padding = data_offset - (tag_table_offset + len(tag_table))
    tag_table += b"\x00" * padding

    profile = bytes(header) + tag_table + tag_data

    assert len(profile) == profile_size, f"{len(profile)} != {profile_size}"
    return profile


# ---------------------------------------------------------------------------
# sRGB ↔ linear conversion
# ---------------------------------------------------------------------------

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Vectorized sRGB gamma → linear."""
    out = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.clip(out, 0.0, 1.0)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Vectorized linear → sRGB gamma."""
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)
    return np.clip(out, 0.0, 1.0)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0–1) to CIE Lab (D50) via XYZ.

    Uses the D50-adapted sRGB matrix (Bradford) so Lab values are relative
    to the ICC PCS illuminant.
    """
    linear = srgb_to_linear(rgb)

    # sRGB (D65) → XYZ (D50) matrix (Bradford-adapted)
    M = np.array([
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733],
    ])
    xyz = linear @ M.T

    # D50 white point
    Xn, Yn, Zn = 0.9642, 1.0000, 0.8251

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx = f(xyz[:, 0] / Xn)
    fy = f(xyz[:, 1] / Yn)
    fz = f(xyz[:, 2] / Zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.column_stack([L, a, b])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_grid(grid_points: int) -> np.ndarray:
    """Generate an (N^3, 3) grid of RGB values in [0, 1].

    ICC lut16Type CLUT ordering: first input channel (R) changes slowest,
    last input channel (B) changes fastest.
    """
    axis = np.linspace(0.0, 1.0, grid_points)
    # R slowest, G middle, B fastest
    grid = np.column_stack([
        np.repeat(axis, grid_points * grid_points),                        # R (slowest)
        np.tile(np.repeat(axis, grid_points), grid_points),                # G (middle)
        np.tile(axis, grid_points * grid_points),                          # B (fastest)
    ])
    return grid


def transform_through_icc(device_rgb: np.ndarray, icc_path: str) -> np.ndarray:
    """Transform device RGB values through an ICC profile to sRGB.

    Uses Pillow/littleCMS to apply the profile.
    """
    from PIL.ImageCms import (
        Intent,
        applyTransform,
        buildTransformFromOpenProfiles,
    )

    n = len(device_rgb)
    # Build a 1-pixel-tall image from the grid values
    # We'll process in chunks to keep memory reasonable
    chunk_size = 1_000_000
    srgb_profile = createProfile("sRGB")
    input_profile = ImageCmsProfile(icc_path)

    result = np.empty_like(device_rgb)

    transform = buildTransformFromOpenProfiles(
        input_profile,
        srgb_profile,
        "RGB",
        "RGB",
        renderingIntent=Intent.RELATIVE_COLORIMETRIC,
    )

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = device_rgb[start:end]
        # Create image from chunk
        pixels = (chunk * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(pixels.reshape(1, -1, 3), "RGB")
        applyTransform(img, transform, inPlace=True)
        out = np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0
        result[start:end] = out

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bake an ICC camera input profile + Hald CLUT into a single ICC input profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    %(prog)s camera.icc fuji400h.png -o camera_fuji400h.icc
    %(prog)s camera.icc portra160.png -o camera_portra.icc -g 33
    %(prog)s camera.icc portra160.png -o camera_portra_half.icc -s 50
    %(prog)s camera.icc bw_look.png -o camera_bw.icc --desc "Camera + B&W Look"
""",
    )
    parser.add_argument("icc", help="Path to the input ICC camera profile")
    parser.add_argument("hald", help="Path to the Hald CLUT PNG image")
    parser.add_argument("-o", "--output", default="combined.icc",
                        help="Output ICC profile path (default: combined.icc)")
    parser.add_argument("-g", "--grid", type=int, default=33,
                        help="CLUT grid points per axis (default: 33, max ~51 for v2 profiles)")
    parser.add_argument("--desc", default=None,
                        help="Profile description (default: auto-generated)")
    parser.add_argument("--copyright", default="Generated by bake_icc_hald.py",
                        help="Copyright string")
    parser.add_argument("-s", "--strength", type=float, default=100.0,
                        help="Strength of the Hald CLUT effect in percent (0=ICC only, 100=full CLUT, default: 100)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress info")
    args = parser.parse_args()

    grid_points = args.grid
    if grid_points < 2:
        parser.error("Grid points must be >= 2")
    if grid_points > 51:
        print(f"Warning: grid size {grid_points} is very large and will produce "
              f"a {grid_points**3 * 6 / 1e6:.1f} MB CLUT. Consider 33 or less.",
              file=sys.stderr)

    strength = args.strength
    if not (0.0 <= strength <= 100.0):
        parser.error("Strength must be between 0 and 100")

    # ---- Step 1: Load the Hald CLUT ----
    if args.verbose:
        print(f"Loading Hald CLUT: {args.hald}")
    clut_data, hald_level = load_hald_clut(args.hald)
    if args.verbose:
        hald_size = hald_level * hald_level
        print(f"  Hald level {hald_level} → {hald_size} nodes/axis ({hald_size**3} entries)")

    # ---- Step 2: Generate device RGB grid ----
    if args.verbose:
        print(f"Generating {grid_points}^3 = {grid_points**3} grid points")
    device_grid = generate_grid(grid_points)

    # ---- Step 3: Transform through existing ICC profile → sRGB ----
    if args.verbose:
        print(f"Transforming through ICC profile: {args.icc}")
    try:
        # Quick check: try to read the profile description
        prof = ImageCmsProfile(args.icc)
        desc = getProfileDescription(prof)
        if args.verbose:
            print(f"  Profile: {desc}")
    except Exception as e:
        print(f"Error reading ICC profile: {e}", file=sys.stderr)
        sys.exit(1)

    srgb_values = transform_through_icc(device_grid, args.icc)
    if args.verbose:
        print(f"  sRGB range: [{srgb_values.min():.4f}, {srgb_values.max():.4f}]")

    # ---- Step 4: Apply Hald CLUT ----
    if args.verbose:
        print("Applying Hald CLUT")
    looked_srgb = apply_hald_clut(srgb_values, clut_data, hald_level)
    if args.verbose:
        print(f"  Output sRGB range: [{looked_srgb.min():.4f}, {looked_srgb.max():.4f}]")

    # ---- Step 4b: Blend according to strength ----
    alpha = strength / 100.0
    if alpha < 1.0:
        if args.verbose:
            print(f"Blending CLUT effect at {strength:.1f}% strength")
        looked_srgb = srgb_values * (1.0 - alpha) + looked_srgb * alpha
        looked_srgb = np.clip(looked_srgb, 0.0, 1.0)

    # ---- Step 5: Convert looked sRGB → Lab (D50) ----
    if args.verbose:
        print("Converting to Lab (D50)")
    lab_values = srgb_to_lab(looked_srgb)
    if args.verbose:
        print(f"  L range: [{lab_values[:, 0].min():.1f}, {lab_values[:, 0].max():.1f}]")
        print(f"  a range: [{lab_values[:, 1].min():.1f}, {lab_values[:, 1].max():.1f}]")
        print(f"  b range: [{lab_values[:, 2].min():.1f}, {lab_values[:, 2].max():.1f}]")

    # ---- Step 6: Build the combined ICC profile ----
    if args.verbose:
        print("Building ICC profile")

    if args.desc is None:
        try:
            orig_desc = getProfileDescription(ImageCmsProfile(args.icc)).strip()
        except Exception:
            orig_desc = "Camera"
        description = f"{orig_desc} {strength:.0f}%"
    else:
        description = args.desc

    profile_bytes = build_icc_profile(
        lab_values,
        grid_points,
        description=description,
        copyright_text=args.copyright,
    )

    with open(args.output, "wb") as f:
        f.write(profile_bytes)

    size_kb = len(profile_bytes) / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB, {grid_points}^3 CLUT, description: \"{description}\")")

    # ---- Step 7: Verify the profile is readable ----
    if args.verbose:
        print("Verifying profile...")
    try:
        test_prof = ImageCmsProfile(args.output)
        test_desc = getProfileDescription(test_prof)
        print(f"  Verification OK: \"{test_desc}\"")
    except Exception as e:
        print(f"  Warning: profile verification failed: {e}", file=sys.stderr)
        print("  The profile may still work in some applications.", file=sys.stderr)


if __name__ == "__main__":
    main()
