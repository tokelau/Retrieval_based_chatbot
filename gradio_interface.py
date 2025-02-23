from inference import get_answer, get_answer_faiss

def get_response(message, history):
    response = get_answer(message)
    confidence = str(round(response[1], 2))
    time = str(round(response[2], 2))
    return f'{response[0]}<br><font color="#dedede" size=0.2>confidence: {confidence}</font><br><font color="#dedede" size=0.2>time: {time}</font>'

def get_response_faiss(message, history):
    response = get_answer_faiss(message)
    confidence = str(round(response[1], 2))
    time = str(round(response[2], 2))
    return f'{response[0]}<br><font color="#dedede" size=0.2>confidence: {confidence}</font><br><font color="#dedede" size=0.2>time: {time}</font>'
    
import gradio as gr

demo = gr.ChatInterface(get_response_faiss, 
                        title="Talk to Dr.House!",
                        theme="soft",)

demo.launch()