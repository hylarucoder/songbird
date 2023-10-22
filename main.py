from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
import gradio as gr
import mdtex2html
import os
import torch

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True).half().cuda()
model = model.eval()

def predict(input):
    query  = tokenizer(input, return_tensors='pt').to(torch.device("cuda"))
    pred = model.generate(**query, max_new_tokens=1024, repetition_penalty=1.1)
    response =tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return response

iface = gr.Interface(fn=predict, inputs="text", outputs="text",title='baichuan7B')
iface.launch(server_name='0.0.0.0',server_port=12098, share=True, inbrowser=True)
