from lib.providers.services import service
import os
import asyncio
import time
from mindroot.lib.utils.backoff import ExponentialBackoff
import base64
from io import BytesIO
from openai import AsyncOpenAI
import json

client = AsyncOpenAI(
       base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

# Backoff managers for different error types
_429_backoff = ExponentialBackoff(initial_delay=1.0, max_delay=30.0, factor=2.0, jitter=True)
_503_backoff = ExponentialBackoff(initial_delay=0.25, max_delay=30.0, factor=2.0, jitter=True)
_MAX_RETRIES = 8

def concat_text_lists(message):
    """Concatenate text lists into a single string"""
    # if the message['content'] is a list
    # then we need to concatenate the list into a single string
    out_str = ""
    if isinstance(message['content'], str):
        return message
    else:
        for item in message['content']:
            if isinstance(item, str):
                out_str += item + "\n"
            else:
                out_str += item['text'] + "\n"
    message.update({'content': out_str})
    return message

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, 
                     temperature=0.0, max_tokens=500, num_gpu_layers=0):
    identifier = f"stream_chat_{model or 'default'}"
    try:
        print("Cerebras stream_chat (OpenAI compatible mode)")
        #max_tokens = 120
        model_name = os.environ.get("AH_OVERRIDE_LLM_MODEL", "llama3.1-8b")
        if model:
            model_name = model
        # look at the last message and the one before that
        # if the role of both of them is the same
        # this is not valid
        # so we need to remove the last message
        #last_role = messages[-1]['role']
        #second_last_role = messages[-2]['role']
        #if last_role == second_last_role:
        #    messages = messages[:-1]

        messages = [concat_text_lists(m) for m in messages]

        # Retry logic with exponential backoff
        for attempt in range(_MAX_RETRIES):
            try:
                # Check if we need to wait before attempting
                wait_429 = _429_backoff.get_wait_time(identifier)
                wait_503 = _503_backoff.get_wait_time(identifier)
                wait_time = max(wait_429, wait_503)
                
                if wait_time > 0:
                    print(f"Cerebras backoff: waiting {wait_time:.2f}s before attempt {attempt + 1}")
                    await asyncio.sleep(wait_time)


                stream = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                print("Opened stream with model:", model_name)
                
                # Record success and reset backoff
                _429_backoff.record_success(identifier)
                _503_backoff.record_success(identifier)
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e)
                print(f'Cerebras error on attempt {attempt + 1}: {e}')
                
                # Check for retryable errors
                if '429' in error_str:
                    _429_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        raise Exception(f"Max retries ({_MAX_RETRIES}) exceeded for 429 error: {e}")
                elif '503' in error_str:
                    _503_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        raise Exception(f"Max retries ({_MAX_RETRIES}) exceeded for 503 error: {e}")
                else:
                    # Non-retryable error, re-raise immediately
                    raise

        async def content_stream(original_stream):
            done_reasoning = False
            async for chunk in original_stream:
                if os.environ.get('AH_DEBUG') == 'True':
                    #    #print('\033[93m' + str(chunk) + '\033[0m', end='')
                    print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content or ""

        return content_stream(stream)

    except Exception as e:
        print('Cerebras error:', e)
        raise

@service()
async def format_image_message(pil_image, context=None):
    """Format image for DeepSeek using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

@service()
async def get_image_dimensions(context=None):
    """Return max supported image dimensions for DeepSeek"""
    return 4096, 4096, 16777216  # Max width, height, pixels


@service()
async def get_service_models(context=None):
    """Get available models for the service"""
    identifier = "get_service_models"
    try:
        # Retry logic with exponential backoff
        for attempt in range(_MAX_RETRIES):
            try:
                # Check if we need to wait before attempting
                wait_429 = _429_backoff.get_wait_time(identifier)
                wait_503 = _503_backoff.get_wait_time(identifier)
                wait_time = max(wait_429, wait_503)
                
                if wait_time > 0:
                    print(f"Cerebras backoff: waiting {wait_time:.2f}s before attempt {attempt + 1}")
                    await asyncio.sleep(wait_time)

                all_models = await client.models.list()
                ids = []
                for model in all_models.data:
                    ids.append(model.id)
                
                # Record success and reset backoff
                _429_backoff.record_success(identifier)
                _503_backoff.record_success(identifier)
                return {'stream_chat': ids}
                
            except Exception as e:
                error_str = str(e)
                print(f'Cerebras models error on attempt {attempt + 1}: {e}')
                
                # Check for retryable errors
                if '429' in error_str:
                    _429_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        return {'stream_chat': []}
                elif '503' in error_str:
                    _503_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        return {'stream_chat': []}
                else:
                    # Non-retryable error, re-raise immediately
                    raise
    except Exception as e:
        return {'stream_chat': []}
