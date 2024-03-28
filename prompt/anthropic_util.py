import anthropic

class Claude():
    def __init__(self, engine, api_key, system_prompt, proxy, max_tokens=300):
        self.engine = engine
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.proxy = proxy
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(
            api_key = self.api_key
        )
        self.messages = []

    def ask(self, question: str):
        self.messages.append({
            'role': 'user',
            'content': question
        })
        response = self.client.messages.create(
            model = self.engine,
            max_tokens = self.max_tokens,
            temperature=0.8,
            system = self.system_prompt,
            messages = self.messages,
        ).messages
        self.messages = response
        return response[-1].content
