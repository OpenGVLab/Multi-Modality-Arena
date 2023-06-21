
import openai
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=0.1, max=60), stop=stop_after_attempt(20))
def call_gpt(chatgpt_messages, model="gpt-3.5-turbo", temp_gpt=0.0):
    success = False
    while not success:
        try:
            #print("chatgpt_messages",chatgpt_messages)
            response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=temp_gpt, max_tokens=512)
            reply = response['choices'][0]['message']['content']
            total_tokens = response['usage']['total_tokens']
            success = True
            return reply, total_tokens
        except Exception as e:
            print('[Worker] an exception occured: %s (%s). retrying in 3 minutes.' % (type(e), str(e)))
            time.sleep(180)