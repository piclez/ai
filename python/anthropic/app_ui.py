import anthropic
import gradio as gr

def query_anthropic_model(user_question):
  client = anthropic.Anthropic()
  message =  client.messages.create(
    model="claude-2.1",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": user_question}
    ]
  )
  return message.content[0].text

iface = gr.Interface(fn=query_anthropic_model,
                     inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
                     outputs="text",
                     title="Anthropic Model Claude 2.1",
                     description="Type your question to get an answer from the Anthropic model."
                     )
iface.launch()
