import anthropic
import httpx

class Claude():
    def __init__(self, engine, api_key, system_prompt, proxy='http://127.0.0.1:7890', max_tokens=300):
        self.engine = engine
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.proxy = proxy
        self.max_tokens = max_tokens
        self.client = anthropic.Client(
            api_key = self.api_key,
            proxies = httpx.Proxy(proxy if isinstance(proxy, str) else 'http://127.0.0.1:7890')
        )
        self.messages = []

    def ask(self, question: str):
        response = self.client.messages.create(
            model = self.engine, # "claude-2.1",
            max_tokens = self.max_tokens,
            system = self.system_prompt,  # <-- system prompt
            messages=[
                {"role": "user", "content": question}  # <-- user prompt
            ]
        )
        return response.content[-1].text
