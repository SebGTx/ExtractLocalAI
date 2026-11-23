from fastapi import FastAPI

from extractlocalai.api.documents import router

app = FastAPI(
    title="ExtractLocalAI"
)

app.include_router(router)
