"""
Scenery asset management: ground/bark textures and a distant-mountain mesh.

Assets are downloaded once into ``<project>/assets`` and reused thereafter, so
the app works offline after the first run. Everything degrades gracefully: if a
download fails (offline, blocked), the corresponding path comes back ``None``
and the visualization falls back to flat colours.

Sources (CC0):
  - Ground grass + tree bark textures: Poly Haven (https://polyhaven.com).
  - Distant mountain: the Mount St. Helens elevation model shipped by the
    VTK data repository, downloaded via ``pyvista.examples`` and baked to a
    local ``.ply`` surface.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Dict, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(_ROOT, "assets")
TEX_DIR = os.path.join(ASSETS_DIR, "textures")

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) turret3d"}
_CTX = ssl.create_default_context()
_TIMEOUT = 60


def _open(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), context=_CTX, timeout=_TIMEOUT
    )


def _download(url: str, dest: str) -> Optional[str]:
    """Fetch ``url`` to ``dest`` (skipping if already present). None on failure."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with _open(url) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[assets] could not download {os.path.basename(dest)}: {exc}")
        if os.path.exists(dest):
            os.remove(dest)
        return None


def _polyhaven_map_url(slug: str, map_key: str, res: str = "1k", fmt: str = "jpg"):
    """Resolve a Poly Haven texture/map download URL from its files manifest."""
    try:
        with _open(f"https://api.polyhaven.com/files/{slug}") as r:
            manifest = json.load(r)
        return manifest[map_key][res][fmt]["url"]
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[assets] manifest lookup failed for {slug}/{map_key}: {exc}")
        return None


def _ensure_texture(dest_name: str, slug: str, map_key: str) -> Optional[str]:
    dest = os.path.join(TEX_DIR, dest_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = _polyhaven_map_url(slug, map_key)
    return _download(url, dest) if url else None


def _ensure_mountain() -> Optional[str]:
    """Download + bake the St. Helens DEM into a local surface mesh."""
    dest = os.path.join(ASSETS_DIR, "mountain_sthelens.ply")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        from pyvista import examples

        dem = examples.download_st_helens()
        surf = dem.warp_by_scalar().extract_geometry().triangulate()
        surf = surf.decimate(0.7)
        surf.save(dest)
        return dest
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[assets] could not build mountain mesh: {exc}")
        return None


def ensure_assets() -> Dict[str, Optional[str]]:
    """Ensure all scenery assets exist locally. Returns a name->path map
    (values are ``None`` when an asset is unavailable)."""
    os.makedirs(TEX_DIR, exist_ok=True)
    assets = {
        "grass": _ensure_texture(
            "aerial_grass_rock_diff_1k.jpg", "aerial_grass_rock", "Diffuse"
        ),
        "bark": _ensure_texture("fir_bark_diff_1k.jpg", "fir_tree_01", "bark_diff"),
        "mountain": _ensure_mountain(),
    }
    return assets
