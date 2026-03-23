from fastapi import Request

async def get_context(request: Request):
    return request.app.state.context

async def get_request_object(request: Request):
    return request