from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "bad_request"):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, resource: str, id_: str):
        super().__init__(f"{resource} '{id_}' not found", 404, "not_found")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("app").exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": str(exc)}},
        )