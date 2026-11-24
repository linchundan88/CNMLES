from openai import OpenAI
import os
import time
from dotenv import load_dotenv
load_dotenv()

# 可重试的异常类型（OpenAI 同步客户端）
RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,                      # 网络层异常（如 Max retries exceeded）
)
try:
    import openai
    RETRYABLE_EXCEPTIONS += (
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIConnectionError,
    )
except Exception:
    pass


def get_llm_client(model_name, timeout=30):

    # if model_name in ['GPT-5.5', 'Gemini-3.5-Flash', 'Qwen3.7-Max', 'doubao-seed-2-0-pro-260215']:
    client = OpenAI(
        base_url=os.getenv("SERVICE_URL_ZZZ"),
        api_key=os.getenv("API_KEY_ZZZ"),
        timeout=timeout
    )

    return client



def process_llm_prediction(chat_client, model_name, system_prompt, user_prompt,
                           temperature=-1, max_completion_tokens=-1, index=-1,
                           max_retries=3, retry_base_delay=1.0):
    """
    调用 LLM 进行预测，内置可配置的出错重试（指数退避）。

    Parameters:
    chat_client: OpenAI 客户端实例
    model_name (str): 模型名称
    system_prompt (str): system prompt
    user_prompt (str): user prompt
    temperature (int): -1 表示不传该参数
    max_completion_tokens (int): -1 表示不传该参数
    index: 多线程模式下携带的索引，原样返回
    max_retries (int): 出错时最大重试次数（不含首次调用），默认 3
    retry_base_delay (float): 指数退避的基础等待秒数，默认 1.0
    """

    # 构建API调用参数
    api_params = {
        'model': model_name,
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': user_prompt}],
    }
    if temperature != -1:
        api_params['temperature'] = temperature
    if max_completion_tokens != -1:
        api_params['max_completion_tokens'] = max_completion_tokens

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            completion = chat_client.chat.completions.create(**api_params)
        except RETRYABLE_EXCEPTIONS as e:
            last_error = e
            if attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                print(f'[retry {attempt + 1}/{max_retries}] LLM call failed: {e}; '
                      f'waiting {delay:.1f}s...')
                time.sleep(delay)
                continue
            # 最后一次重试仍失败
            error_msg = f"System Error: {e} (after {max_retries} retries)"
            return error_msg if index == -1 else (index, error_msg)
        except Exception as e:
            # 不可重试的异常（如 401 AuthenticationError、400 BadRequestError 等），直接返回错误
            error_msg = f"System Error: {e}"
            return error_msg if index == -1 else (index, error_msg)

        if completion.choices is None:
            error_msg = 'System Error: completion.choices is None'
            return error_msg if index == -1 else (index, error_msg)

        prediction = completion.choices[0].message.content
        if prediction == '':
            # max_completion_tokens 过小时 prediction 为空字符串，视为可重试
            last_error = RuntimeError('empty completion content')
            if attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                print(f'[retry {attempt + 1}/{max_retries}] empty prediction; '
                      f'waiting {delay:.1f}s...')
                time.sleep(delay)
                continue
            error_msg = 'System Error: empty str (after retries)'
            return error_msg if index == -1 else (index, error_msg)

        if index == -1:
            return prediction
        else:
            return index, prediction

    # 理论上到不了这里，但兜底
    error_msg = f"System Error: exhausted retries, last error: {last_error}"
    return error_msg if index == -1 else (index, error_msg)

    