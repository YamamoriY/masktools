from lib.llmclient.chatgpt import ChatGPT


def main() -> None:
    client = ChatGPT()
    answer = client.ask(
        text="この画像に写っているものを日本語で簡潔に説明してください。",
        images=["data/testdata/input_03.jpg"],
    )
    print(answer)


if __name__ == "__main__":
    main()
