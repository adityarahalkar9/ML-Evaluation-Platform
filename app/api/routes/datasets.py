from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_dataset_service
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", status_code=201)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file"),
    target_column: str = Form(..., description="Name of the label column"),
    service: DatasetService = Depends(get_dataset_service),
):
    content = await file.read()
    return service.ingest(file.filename or "upload.csv", content, target_column)


@router.get("")
def list_datasets(service: DatasetService = Depends(get_dataset_service)):
    return service.list()


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, service: DatasetService = Depends(get_dataset_service)):
    return service.get(dataset_id)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, force: bool = False,
                   service: DatasetService = Depends(get_dataset_service)):
    return service.delete(dataset_id, force=force)