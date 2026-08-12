def main() -> None:
    import uvicorn

    uvicorn.run("bookchat.api.app:app", host="0.0.0.0", port=8000, reload=True)
