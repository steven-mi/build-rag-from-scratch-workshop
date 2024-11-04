import os
from mistralai import Mistral


class LargeLanguageModel(object):
    def __init__(self, model="mistral-small-latest"):
        self.model = model
        api_key = os.environ.get("MISTRAL_API_KEY", None)
        if api_key is None:
            raise Exception(
                f"`MISTRAL_API_KEY` is None. Please set it in your environment variables."
            )
        self.client = Mistral(api_key=api_key)

    def call(self, prompt):
        chat_response = self.client.chat.complete(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return chat_response.choices[0].message.content


if __name__ == "__main__":
    model = LargeLanguageModel()
    print(model.call("Who is the president of Germany"))
