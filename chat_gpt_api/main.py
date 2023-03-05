from chat import Chat

if __name__ == '__main__':
    with open("api.key", "r") as f:
        API_KEY = f.read()

    gpt = Chat(API_KEY, "Be a programmer")

    while(question := input("\n")) != "exit":
        answer = gpt.ask(question)
        print(answer)