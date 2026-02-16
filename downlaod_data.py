# importing the necessary libraries
# this one require astroquery tool ( if not then: pip install astroquery)

import numpy as np
import os
import subprocess
import warnings

from astroquery.heasarc import Heasarc
from astropy.table import Table
from astropy.units import UnitsWarning

warnings.simplefilter("ignore", UnitsWarning)

CATALOG_FILE = "gbm_catalog.fits"

BASE_GBM = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/"             # url for the FERMI data
BASE_LAT = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/triggers/"


# --------------------------------------------------
# Download GBM catalog once ( to get the trigger name)
# --------------------------------------------------
def ensure_catalog():

    if not os.path.exists(CATALOG_FILE):

        print("Downloading GBM burst catalog (one-time setup)...")

        h = Heasarc()
        h.ROW_LIMIT = 50000  # unlimited   ### this is still not working and only returning 1000 rows/1000 GRBs 

        tbl = h.query_region(
            "0 0",
            mission="fermigbrst",
            radius="180 deg"
        )

        print("Catalog columns:", tbl.colnames)  # helps debugging once

        tbl.write(CATALOG_FILE, overwrite=True)



# --------------------------------------------------
# GRB name → trigger
# --------------------------------------------------
def get_trigger_from_name(grb_name):
    
    print(f"grb name {grb_name}")
    
    ensure_catalog()
    catalog = Table.read(CATALOG_FILE)

    # GRB240825A → 240825
    date_code = grb_name[3:9]
    year_code = date_code[0:2]    # to get the directory inside the catalouge
    

    for name in catalog["NAME"]:

        name_str = str(name)

        # Example catalog entry: GRB240825662
        if name_str.startswith("GRB" + date_code):

            trigger_num = name_str.replace("GRB", "")
            return f"bn{trigger_num}", f"20{year_code}"   # this will print GRB name as trigger id (for e.g. GRB240825A ----bn240825667)
	
    raise ValueError(f"No trigger found for {grb_name}")


# --------------------------------------------------
# Download directory
# --------------------------------------------------
def wget_download(url, outdir):

    os.makedirs(outdir, exist_ok=True)

    print(f"\nDownloading:\n{url}")

    cmd = [
        "wget",
        "-r",
        "-np",
        "-nH",
        "--cut-dirs=7",
        "-e robots=off",
        "-R", "index.html*",
        "-P", outdir,
        url
    ]

    subprocess.run(cmd)


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def download_grb(grb_name):

    trigger,year = get_trigger_from_name(grb_name)

    print(f"\nResolved {grb_name} → {trigger}")

    gbm_url = f"{BASE_GBM}{year}/{trigger}/current/"    
    lat_url = f"{BASE_LAT}{year}/{trigger}/current/"
    print(f" gbm url {gbm_url}")
    wget_download(gbm_url, f"DATA/{grb_name}/GBM")
    wget_download(lat_url, f"DATA/{grb_name}/LAT_LLE")


# --------------------------------------------------
if __name__ == "__main__":

    name = input("Enter GRB name (e.g. GRB240825A): ").strip()
    download_grb(name)

