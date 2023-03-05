import openai


class Chat:
    def __init__(self, api_key, role):
        openai.api_key = api_key
        self.dialogue = [{"role": "system", "content": role}]

    def ask(self, question):
        self.dialogue.append({"role": "user", "content": question})
        result = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.dialogue)
        answer = result.choices[0].message.content
        self.dialogue.append({"role": "assistant", "content": answer})
        return answer
