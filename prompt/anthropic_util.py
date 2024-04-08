import anthropic
import httpx, base64


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Claude():
    def __init__(self, engine, api_key, system_prompt, proxy='http://127.0.0.1:7890', max_tokens=300):
        self.engine = engine
        self.api_key = api_key
        self.system_prompt = system_prompt + '\nAny irrelevant characters appear in your response is STRICTLY forbidden. '
        self.proxy = proxy
        self.max_tokens = max_tokens
        self.client = anthropic.Client(
            api_key = self.api_key,
            proxies = httpx.Proxy(proxy if isinstance(proxy, str) else 'http://127.0.0.1:7890')
        )
        self.messages = []
    def pre_cut(self, prompt):
        return prompt[prompt.find('('):]
    def ask(self, question: str):
        response = self.client.messages.create(
            model = self.engine, # "claude-2.1",
            max_tokens = self.max_tokens,
            system = self.system_prompt,  # <-- system prompt
            messages=[
                {"role": "user", "content": question}  # <-- user prompt
            ]
        ).content[-1].text
        return response

class Vision_Claude(Claude):
    def __init__(self, engine, api_key, proxy='http://127.0.0.1:7890', max_tokens=1024):
        super().__init__(engine, api_key, system_prompt='', proxy=proxy, max_tokens=max_tokens)

    def ask(self, question: str, encoded_img):
        response = self.client.messages.create(
            model = self.engine,
            max_token = self.max_token,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_img,
                            },
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]
        ).content[-1].text
        return response


Question = lambda x, y, w, h: f"I want to edit this image with an instruction \"{x} in the image\", please arrange the location of \"{y}\". "\
                        f"And you should tell me the of \"{y}\", in form of $(x,y,w,h)$, where $x,y$ indicates the coordinates "\
                        f"and $(w,h)$ indicates the width and height. The image sized {(w,h)}. " + \
                        "Note that you are getting a ratio, so you only need to output a ratio too. " if (w,h) == (1,1) else ""


def ask_claude_vision(img_encoded, agent, edit_txt, target, img_size):
    question = Question(edit_txt, target, img_size)
    response = agent.ask(question, img_encoded)
    return response

def claude_vision_box(opt, target_noun: str, size):
    img_encoded = encode_image(opt.in_dir)
    agent = Vision_Claude(engine=opt.engine, api_key=opt.api_key)

    raw_return = ask_claude_vision(img_encoded, agent, opt.edit_txt, target_noun, size) # "(x, y, w, h)" string return expected
    raw_return = raw_return[raw_return.find('('): raw_return.rfind(')')+1]

    return f'[{target_noun}, {raw_return}]'




