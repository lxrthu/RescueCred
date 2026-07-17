#!/usr/bin/env python3
from rescuecredit.azure_client import AzureOpenAIAdapter


def main() -> None:
    adapter = AzureOpenAIAdapter()
    answer = adapter.complete(
        [{"role": "system", "content": [{"type": "text", "text": "你是一个帮助用户查找信息的 AI 助手。"}]}, {"role": "user", "content": "只回复 AZURE_OK"}],
        max_tokens=16,
        temperature=0.0,
    )
    print(answer)


if __name__ == "__main__":
    main()

