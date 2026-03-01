"""
Zotero API client for listing libraries/groups and fetching attachment files.
Only attachments with allowed file types are returned for sync.
"""
import os
import io
import requests

# File extensions this system can ingest (must match routes.extract_text_from_file)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_EXTENSIONS_LABEL = "PDF, DOCX, TXT"

ZOTERO_API_BASE = "https://api.zotero.org"
DEFAULT_HEADERS = {"Zotero-API-Version": "3"}


def _headers(api_key):
    return {**DEFAULT_HEADERS, "Authorization": f"Bearer {api_key}"}


def _get(url, api_key, timeout=30):
    r = requests.get(url, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    return r


def is_allowed_extension(filename):
    """Return True if filename has an allowed extension (case-insensitive)."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_user_groups(user_id, api_key):
    """
    Fetch groups the user belongs to (shared libraries).
    Returns list of {"id", "name", "type": "group"}.
    """
    url = f"{ZOTERO_API_BASE}/users/{user_id}/groups"
    r = _get(url, api_key)
    data = r.json()
    return [
        {"id": str(g["id"]), "name": g.get("data", {}).get("name", f"Group {g['id']}"), "type": "group"}
        for g in data
    ]


def get_libraries(user_id, api_key):
    """
    Return list of libraries: My Library + all groups.
    Each item: {"id", "name", "type": "user" | "group"}.
    """
    libraries = [
        {"id": str(user_id), "name": "My Library", "type": "user"}
    ]
    groups = get_user_groups(user_id, api_key)
    libraries.extend(groups)
    return libraries


def get_items_with_attachments(library_type, library_id, api_key):
    """
    Fetch all items (with pagination), then for each item get children (attachments).
    Yields dicts: {"item_key", "filename", "contentType", "linkMode", "parent_key", "library_type", "library_id"}.
    Only includes attachments with allowed file extensions; others are skipped.
    """
    if library_type == "user":
        base = f"{ZOTERO_API_BASE}/users/{library_id}"
    else:
        base = f"{ZOTERO_API_BASE}/groups/{library_id}"

    seen_attachment_keys = set()
    start = 0
    limit = 100

    while True:
        url = f"{base}/items?start={start}&limit={limit}"
        r = _get(url, api_key)
        items = r.json()
        if not items:
            break

        for item in items:
            data = item.get("data", {})
            key = data.get("key")
            item_type = data.get("itemType", "")
            if item_type == "attachment":
                link_mode = data.get("linkMode", "")
                if link_mode not in ("imported_file", "imported_url", "linked_file"):
                    continue
                filename = data.get("filename") or data.get("title") or "document"
                if not is_allowed_extension(filename) or key in seen_attachment_keys:
                    continue
                seen_attachment_keys.add(key)
                yield {
                    "item_key": key,
                    "filename": filename,
                    "contentType": data.get("contentType", ""),
                    "linkMode": link_mode,
                    "parent_key": None,
                    "library_type": library_type,
                    "library_id": library_id,
                }
            else:
                # Get children (attachments) for this item
                try:
                    cr = _get(f"{base}/items/{key}/children", api_key)
                    children = cr.json()
                except Exception:
                    children = []
                for child in children:
                    cdata = child.get("data", {})
                    ckey = cdata.get("key")
                    if not ckey or ckey in seen_attachment_keys:
                        continue
                    link_mode = cdata.get("linkMode", "")
                    if link_mode not in ("imported_file", "imported_url", "linked_file"):
                        continue
                    filename = cdata.get("filename") or cdata.get("title") or "document"
                    if not is_allowed_extension(filename):
                        continue
                    seen_attachment_keys.add(ckey)
                    yield {
                        "item_key": ckey,
                        "filename": filename,
                        "contentType": cdata.get("contentType", ""),
                        "linkMode": link_mode,
                        "parent_key": key,
                        "library_type": library_type,
                        "library_id": library_id,
                    }

        if len(items) < limit:
            break
        start += limit


def download_attachment_file(library_type, library_id, item_key, api_key, timeout=60):
    """
    Download attachment file bytes. Follows redirects (Zotero redirects to S3).
    Returns (bytes, filename) or (None, None) on failure.
    """
    if library_type == "user":
        base = f"{ZOTERO_API_BASE}/users/{library_id}"
    else:
        base = f"{ZOTERO_API_BASE}/groups/{library_id}"
    url = f"{base}/items/{item_key}/file"
    try:
        r = requests.get(url, headers=_headers(api_key), timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        # Filename might be in Content-Disposition
        filename = None
        cd = r.headers.get("Content-Disposition")
        if cd and "filename=" in cd:
            part = cd.split("filename=")[-1].strip().strip('"\'')
            if part:
                filename = part
        if not filename:
            # Fallback: get from item metadata
            meta_url = f"{base}/items/{item_key}"
            mr = _get(meta_url, api_key)
            meta = mr.json()
            filename = meta.get("data", {}).get("filename") or meta.get("data", {}).get("title") or "document"
        return (r.content, filename)
    except Exception as e:
        print(f"Zotero download error for {item_key}: {e}")
        return (None, None)


def file_like_for_zotero(data, filename):
    """Wrap bytes and filename in a file-like object compatible with extract_text_from_file."""
    class BytesFileLike:
        def __init__(self, data_bytes, name):
            self._io = io.BytesIO(data_bytes)
            self.filename = name

        def read(self, *args, **kwargs):
            return self._io.read(*args, **kwargs)

        def seek(self, *args, **kwargs):
            return self._io.seek(*args, **kwargs)

    obj = BytesFileLike(data, filename)
    obj.seek(0)
    return obj
