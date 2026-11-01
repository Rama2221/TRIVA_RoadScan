"""HTTP endpoints for report submission, queue access and maintenance review."""
import os

import tempfile

from contextlib import asynccontextmanager

from pathlib import Path


from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pydantic import BaseModel


from .service import RoadScanService
from .settings import load_settings


@asynccontextmanager

async def lifespan(app):

    app.state.service = RoadScanService(

        os.environ.get('ROADSCAN_PACKAGE', '.'),

        os.environ.get('ROADSCAN_STATE', './app_state'),

        device=os.environ.get('ROADSCAN_DEVICE', 'cpu'))

    yield


app = FastAPI(title='RoadScan', lifespan=lifespan)


class Review(BaseModel):

    confirm: bool


@app.post('/reports')

def submit(image: UploadFile = File(...), latitude: float | None = Form(None),

           longitude: float | None = Form(None), report_id: str | None = Form(None)):

    with tempfile.TemporaryDirectory() as folder:

        path = Path(folder)/'upload'

        with path.open('wb') as output:

            while chunk := image.file.read(1024*1024):

                output.write(chunk)

        try:

            return app.state.service.submit_report(path, latitude, longitude, report_id)

        except ValueError as error:

            raise HTTPException(status_code=422, detail=str(error)) from error


@app.get('/reports')

def queue(include_closed: bool = False):

    return app.state.service.list_reports(include_closed=include_closed)


@app.get('/reports/{report_id}')

def get_report(report_id: str):

    try:

        return app.state.service.get_report(report_id)

    except KeyError as error:

        raise HTTPException(status_code=404, detail='Unknown report') from error


@app.post('/reports/{report_id}/duplicate-review')

def review(report_id: str, body: Review):

    try:

        return app.state.service.review_duplicate(report_id, body.confirm)

    except ValueError as error:

        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post('/reports/{report_id}/resolve')

def resolve(report_id: str):

    try:

        return app.state.service.resolve_report(report_id)

    except ValueError as error:

        raise HTTPException(status_code=409, detail=str(error)) from error
