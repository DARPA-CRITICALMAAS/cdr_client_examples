import argparse
import asyncio
import atexit
import hashlib
import hmac
import os
import uuid
from typing import Any

import httpx
import ngrok
import rasterio as rio
import rasterio.transform as riot
import uvicorn
import uvicorn.logging
from cdr_schemas.events import Event, MapEventPayload
from cdr_schemas.georeference import (GeoreferenceResult, GeoreferenceResults,
                                      ProjectionResult)
from cdr_schemas.feature_results import FeatureResults
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException, Request,
                     status)
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pyproj import Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject

Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--cog_id")
args = parser.parse_args()


class Settings(BaseSettings):
    # TO BE CHANGED BY TA1-4 system.
    system_name: str = "xcorp_georeferencer"
    system_version: str = "0.0.1"
    ml_model_name: str = "xcorp_georef_model"
    ml_model_version: str = "0.0.1"

    # Local port to run on
    local_port: int = 9999
    # To be filled in programmatically via ngrok below.
    callback_url: str = ""
    # Secret string used for signature verification on callback.  Changed by TA1-4 system.
    registration_secret: str = "mysecret"

    # To be provided to TA1-4 system by CDR admin
    user_api_token: str = os.environ["CDR_API_TOKEN"]
    cdr_host: str = "https://api.cdr.land"

    # To be filled in programmatically after registration process below.  Needed to remove registration.
    registration_id: str = ""

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


app_settings = Settings()

# Get ngrok to give us an endpoint
listener = ngrok.forward(app_settings.local_port, authtoken_from_env=True)
app_settings.callback_url = listener.url() + "/hook"


def clean_up():
    # delete our registered system at CDR on program end
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    client.delete(f"{app_settings.cdr_host}/user/me/register/{
                  app_settings.registration_id}", headers=headers)


# register clean_up
atexit.register(clean_up)

app = FastAPI()

# GIS helpers.  NOT CDR related.
#
#


def project_(raw_path, pro_cog_path, geo_transform, crs):
    with rio.open(raw_path) as raw:
        bounds = riot.array_bounds(raw.height, raw.width, geo_transform)
        pro_transform, pro_width, pro_height = calculate_default_transform(
            crs, crs, raw.width, raw.height, *tuple(bounds)
        )
        pro_kwargs = raw.profile.copy()
        pro_kwargs.update(
            {
                "driver": "COG",
                "crs": {"init": crs},
                "transform": pro_transform,
                "width": pro_width,
                "height": pro_height,
            }
        )
        _raw_data = raw.read()
        with rio.open(pro_cog_path, "w", **pro_kwargs) as pro:
            for i in range(raw.count):
                _ = reproject(
                    source=_raw_data[i],
                    destination=rio.band(pro, i + 1),
                    src_transform=geo_transform,
                    src_crs=crs,
                    dst_transform=pro_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                    num_threads=8,
                    warp_mem_limit=256,
                )


def cps_to_transform(cps, height, to_crs):
    cps = [
        {
            "row": height - float(cp["px_geom"]["rows_from_top"]),
            "col": float(cp["px_geom"]["columns_from_left"]),
            "x": float(cp["map_geom"]["longitude"]),
            "y": float(cp["map_geom"]["latitude"]),
            "crs": cp["crs"],
        }
        for cp in cps
    ]
    cps_p = []
    for cp in cps:
        proj = Transformer.from_crs(cp["crs"], to_crs, always_xy=True)
        x_p, y_p = proj.transform(xx=cp["x"], yy=cp["y"])
        cps_p.append(
            riot.GroundControlPoint(row=cp["row"], col=cp["col"], x=x_p, y=y_p)
        )

    return riot.from_gcps(cps_p)


def dummy_GCPs():
    all_gcps = []

    px = [
        [-111.75, 31.75],
        [-111.75, 32.0],
        [-111.5, 31.75],
        [-111.5, 32.0]
    ]

    locs = [
        [500.41882453833244, 1160.080579652913],
        [496.8866032213555, 7984.080453879687],
        [6281.816662441088, 1161.25163544084],
        [6288.542440280515, 7989.13747091101]
    ]

    for i, x in enumerate(px):
        all_gcps.append(
            {
                "gcp_id": str(uuid.uuid4()),
                "map_geom": {
                    "type": "Point",
                    "longitude": x[0],
                    "latitude": x[1]
                },
                "px_geom": {
                    "type": "Point",
                    "rows_from_top": locs[i][1],
                    "columns_from_left": locs[i][0]
                },
                "confidence": None,
                "model": app_settings.ml_model_name,
                "model_version": app_settings.ml_model_version,
                "crs": "EPSG:4267"
            }
        )
    return all_gcps


def dummy_georeference_result(gcps, cog_id):
    img = Image.open(f"{cog_id}.cog.tif")
    width, height = img.size

    proj_file_name = f"{cog_id}.pro.cog.tif"

    geo_transform = cps_to_transform(gcps, height=height, to_crs="EPSG:4267")

    project_(f"{cog_id}.cog.tif", proj_file_name,
             geo_transform, "EPSG:4267")

    gcp_ids = list(map(lambda x: x["gcp_id"], gcps))
    pr = ProjectionResult(crs="EPSG:4267", gcp_ids=gcp_ids,
                          file_name=proj_file_name)
    gr = GeoreferenceResult(
        likely_CRSs=["EPSG:4267"], map_area=None, projections=[pr])
    return gr

    # project the image


async def georeference_map(req: MapEventPayload,  response_model=GeoreferenceResults):
    cog_id = req['cog_id']
    cog_url = req['cog_url']
    print(f"COG ID: {cog_id}")
    print(f"COG URL: {cog_url}")

    # Download .cog
    print("Downloading...")
    if not os.path.exists(f"{cog_id}.cog.tif"):
        r = httpx.get(cog_url, timeout=1000)
        with open(f"{cog_id}.cog.tif", "wb") as f:
            f.write(r.content)

    print("Creating georef result...")
    gcps = dummy_GCPs()
    result = dummy_georeference_result(gcps, cog_id)

    files_ = []
    files_.append(
        ("files", (result.projections[0].file_name, open(result.projections[0].file_name, "rb"))))

    results = GeoreferenceResults(**{
        "cog_id": cog_id,
        "gcps": gcps,
        "georeference_results": [result],
        "system": app_settings.system_name,
        "system_version": app_settings.system_version
    })

    print("Sending Result to CDR...")
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    resp = client.post(f"{app_settings.cdr_host}/v1/maps/publish/georef",
                       data={"georef_result": results.model_dump_json(exclude_none=True)}, files=files_, headers=headers)
    print("Done!")
    os.remove(result.projections[0].file_name)
    os.remove(f"{cog_id}.cog.tif")
    return results


def dummy_feature_results(cog_id):
    polygon_feature_1 = {
        "id": "1712",
        "crs": "EPSG:4326",
        "cdr_projection_id": "",
        "map_unit": {
            "age_text": "",
            "b_age": None,
            "b_interval": "",
            "lithology": "",
            "name": "",
            "t_age": None,
            "t_interval": "",
            "comments": ""
        },
        "abbreviation": '',
        "label": '',
        "legend_bbox": [0, 0, 0, 0],
        "description": '',
        "pattern": 'solid',
        "color": '#dfa566',
        "category": '',
        "polygon_features": {
            "features": [
                {
                    "id": "polygon_feature_1",
                    "geometry": {
                        "coordinates": [[[-92.3368977795084, 47.96525347233585], [-92.3368977795084, 47.96520426896563], [-92.33684857613818, 47.96520426896563], [-92.33684857613818, 47.96525347233585], [-92.3368977795084, 47.96525347233585]]]
                    },
                    "properties": {
                        "model": "",
                        "model_version": "None",
                        "confidence": .5,
                    }
                }
            ]
        }
    }

    polygon_feature_2 = {
        "id": "1713",
        "crs": "EPSG:4326",
        "cdr_projection_id": "",
        "map_unit": {
            "age_text": "",
            "b_age": None,
            "b_interval": "",
            "lithology": "",
            "name": "",
            "t_age": None,
            "t_interval": "",
            "comments": ""
        },
        "abbreviation": '',
        "label": '',
        "legend_bbox": [0, 0, 0, 0],
        "description": '',
        "pattern": 'solid',
        "color": '#dfa566',
        "category": '',
        "polygon_features": {
            "features": [
                {
                    "id": "polygon_feature_1",
                    "geometry": {
                        "coordinates": [[[-92.01008899446106, 47.944981683802354], [-92.01008899446106, 47.94493248043213], [-92.01003979109083, 47.94493248043213], [-92.01003979109083, 47.944981683802354], [-92.01008899446106, 47.944981683802354]]]
                    },
                    "properties": {
                        "model": "",
                        "model_version": "None",
                        "confidence": .5,
                    }
                }
            ]
        }
    }

    feature_payload = FeatureResults(
        cog_id=cog_id,
        polygon_feature_results=[polygon_feature_1, polygon_feature_2],
        line_feature_results=[],
        point_feature_results=[],
        cog_area_extractions=[],
        cog_metadata_extractions=[],
        system=app_settings.system_name,
        system_version=app_settings.system_version
    )

    return feature_payload


async def post_feature_results(cog_id: str, cog_url: str):

    # Download .cog
    print("Downloading...")
    if not os.path.exists(f"{cog_id}.cog.tif"):
        r = httpx.get(cog_url, timeout=1000)
        with open(f"{cog_id}.cog.tif", "wb") as f:
            f.write(r.content)

    payload = dummy_feature_results(cog_id)

    print("Sending Feature to CDR...")
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    resp = client.post(f"{app_settings.cdr_host}/v1/maps/publish/features",
                       data=payload.model_dump_json(), headers=headers)
    print("Posted Features to CDR!")


async def get_feature_result(id: str):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)
    resp = client.get(f"{app_settings.cdr_host}/v1/maps/extractions/{id}",
                      headers=headers)
    return resp.json()


async def get_georef_result(id: str):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)
    resp = client.get(f"{app_settings.cdr_host}/v1/maps/georef/{id}",
                      headers=headers)
    return resp.json()


async def get_all_extraction_results(cog_id: str):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)
    resp = client.get(f"{app_settings.cdr_host}/v1/maps/cog/{cog_id}/results",
                      headers=headers)
    result_json = resp.json()
    print(f"Got all extraction results! {result_json}")
    return result_json


async def event_handler(evt: Event):
    try:
        match evt:
            case Event(event="ping"):
                print("Received PING!")
            case Event(event="map.process"):
                print("Received MAP!")
                await post_feature_results(cog_id=evt.payload['cog_id'], cog_url=f"https://s3.amazonaws.com/public.cdr.land/cogs/{evt.payload['cog_id']}.cog.tif")
                await georeference_map(evt.payload)
            case Event(event="feature.process"):
                print("Received FEATURE Result!")
                print(evt.payload)
                feature = await get_feature_result(evt.payload["id"])
                print(f"Got feature from event: {feature}")
            case Event(event="georef.process"):
                print("Received GEOREF Result!")
                print(evt.payload)
                georef = await get_georef_result(evt.payload["id"])
                print(f"Got georef from event: {georef}")
            case _:
                print("Nothing to do for event: %s", evt)

    except Exception:
        print("background processing event: %s", evt)
        raise

cdr_signiture = APIKeyHeader(name="x-cdr-signature-256")


async def verify_signature(request: Request, signature_header: str = Depends(cdr_signiture)):

    payload_body = await request.body()
    if not signature_header:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="x-hub-signature-256 header is missing!")

    hash_object = hmac.new(app_settings.registration_secret.encode(
        "utf-8"), msg=payload_body, digestmod=hashlib.sha256)
    expected_signature = hash_object.hexdigest()
    if not hmac.compare_digest(expected_signature, signature_header):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Request signatures didn't match!")

    return True


@app.post("/hook")
async def hook(
    evt: Event,
    background_tasks: BackgroundTasks,
    request: Request,
    verified_signature: bool = Depends(verify_signature),
):
    """Our main entry point for CDR calls"""

    background_tasks.add_task(event_handler, evt)
    return {"ok": "success"}


def run():
    """Run our web hook"""
    uvicorn.run("__main__:app", host="0.0.0.0",
                port=app_settings.local_port, reload=False)


def register_system():
    """Register our system to the CDR using the app_settings"""
    global app_settings
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}

    registration = {
        "name": app_settings.system_name,
        "version": app_settings.system_version,
        "callback_url": app_settings.callback_url,
        "webhook_secret": app_settings.registration_secret,
        # Leave blank if callback url has no auth requirement
        "auth_header": "",
        "auth_token": "",
        # Registers for ALL events
        "events": []

    }

    client = httpx.Client(follow_redirects=True)

    r = client.post(f"{app_settings.cdr_host}/user/me/register",
                    json=registration, headers=headers)

    # Log our registration_id such we can delete it when we close the program.
    app_settings.registration_id = r.json()["id"]


if __name__ == "__main__":
    if args.mode == 'host':
        register_system()
        run()
    if args.mode == 'process':
        asyncio.run(georeference_map(
            {"cog_id": args.cog_id, "cog_url": f"https://s3.amazonaws.com/public.cdr.land/cogs/{args.cog_id}.cog.tif"}))
        asyncio.run(post_feature_results(cog_id=args.cog_id,
                    cog_url=f"https://s3.amazonaws.com/public.cdr.land/cogs/{args.cog_id}.cog.tif"))
        asyncio.run(get_all_extraction_results(cog_id=args.cog_id))
