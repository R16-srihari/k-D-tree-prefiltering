"""
Space-Track.org Satellite Data Extractor & Propagator
Extracts GP class data and propagates all satellites to the first epoch
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

from sgp4.api import Satrec, jday
from spacetrack import SpaceTrackClient


def authenticate():
    """Authenticate with Space-Track.org using credentials from space-track_account.json"""
    account_path = Path(__file__).parent / "space-track_account.json"
    with open(account_path, "r") as f:
        credentials = json.load(f)
    username = credentials["username"]
    password = credentials["password"]
    return SpaceTrackClient(identity=username, password=password)


def fetch_catalog(st):
    """Fetch satellite catalog data and save raw CSV"""
    print("Fetching satellite catalog from Space-Track.org...")

    data = st.gp(
        epoch='>now-10',
        mean_motion='>11.25',
        eccentricity='<0.25',
        decay_date='null-val',
        orderby='epoch desc',
        limit=5000,
        format='csv'
    )

    # Save raw CSV to data directory
    catalog_path = Path(__file__).parent / "space-track_catalog.csv"
    with open(catalog_path, "w", newline='', encoding='utf-8') as f:
        f.write(data)

    # Parse CSV
    lines = data.strip().split('\n')
    reader = csv.DictReader(lines)
    catalog = list(reader)

    print(f"✓ Retrieved {len(catalog)} satellites")
    return catalog


def propagate_to_epoch(sat, target_epoch):
    """Propagate satellite to target epoch using SGP4.

    Uses TLE_LINE1 and TLE_LINE2 directly from the Space-Track CSV.
    Falls back to the satellite's own epoch if the target epoch fails.
    Returns None only if both attempts fail.
    """

    def _try_propagate(epoch, tle_line1, tle_line2):
        try:
            satellite = Satrec.twoline2rv(tle_line1, tle_line2)
            jd, fr = jday(epoch.year, epoch.month, epoch.day,
                          epoch.hour, epoch.minute,
                          epoch.second + epoch.microsecond / 1_000_000)
            e, r, v = satellite.sgp4(jd, fr)
            if e != 0 or e is None:
                return None, e
            if any(math.isnan(c) or math.isinf(c) for c in r + v):
                return None, -1  # NaN/Inf guard
            return {
                'NORAD_CAT_ID': sat.get('NORAD_CAT_ID', ''),
                'OBJECT_NAME': sat.get('OBJECT_NAME', ''),
                'EPOCH': epoch.strftime('%Y-%m-%d %H:%M:%S'),
                'X': r[0], 'Y': r[1], 'Z': r[2],
                'VX': v[0], 'VY': v[1], 'VZ': v[2]
            }, 0
        except Exception as exc:
            return None, f"exception: {exc}"

    tle_line1 = sat.get('TLE_LINE1', '').strip()
    tle_line2 = sat.get('TLE_LINE2', '').strip()

    if not tle_line1 or not tle_line2:
        name = sat.get('OBJECT_NAME', 'Unknown')
        print(f"  Warning: Missing TLE lines for {name}")
        return None

    # Primary attempt: target epoch
    result, err = _try_propagate(target_epoch, tle_line1, tle_line2)
    if result is not None:
        return result

    # Fallback: satellite's own epoch
    sat_epoch_str = sat.get('EPOCH', '')
    if sat_epoch_str:
        try:
            sat_epoch = datetime.fromisoformat(sat_epoch_str)
            result, err = _try_propagate(sat_epoch, tle_line1, tle_line2)
            if result is not None:
                return result
        except Exception:
            pass

    name = sat.get('OBJECT_NAME', 'Unknown')
    if isinstance(err, int):
        print(f"  Warning: Failed to propagate {name}: SGP4 error code {err}")
    else:
        print(f"  Warning: Failed to propagate {name}: {err}")
    return None


def main():
    # Authenticate
    st = authenticate()

    # Fetch catalog
    catalog = fetch_catalog(st)

    if not catalog:
        print("No satellites found")
        return

    # Get target epoch (first/most recent in list)
    target_epoch = datetime.fromisoformat(catalog[0]['EPOCH'])
    print(f"\nPropagating all satellites to epoch: {target_epoch}")

    # Propagate all satellites
    propagated = []
    failed = 0
    for idx, sat in enumerate(catalog, 1):
        result = propagate_to_epoch(sat, target_epoch)
        if result:
            propagated.append(result)
        else:
            failed += 1
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(catalog)}")

    # Save to CSV in the data directory
    if propagated:
        output_dir = Path(__file__).parent.parent  # rso_kdtree/data/
        filename = output_dir / "satellite_propagated.csv"
        keys = propagated[0].keys()

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(propagated)

        print(f"\n✓ Successfully propagated {len(propagated)} satellites ({failed} failed)")
        print(f"✓ Saved to {filename}")
        print(f"\nSample output:")
        print(f"  Satellite: {propagated[0]['OBJECT_NAME']}")
        print(f"  Epoch: {propagated[0]['EPOCH']}")
        print(f"  Position (km): X={propagated[0]['X']:.3f}, Y={propagated[0]['Y']:.3f}, Z={propagated[0]['Z']:.3f}")
    else:
        print("No satellites could be propagated")


if __name__ == "__main__":
    main()
