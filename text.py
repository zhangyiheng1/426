"""
author:张怡恒
"""
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

#使用url直接下载时，将 huggingface.co 直接替换为本站域名hf-mirror.com。使用浏览器或者 wget -c、curl -L、aria2c 等命令行方式即可。
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)