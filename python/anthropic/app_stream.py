import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
  model="claude-2.1",
  max_tokens=1024,
  messages=[
    {"role": "user", "content": "Give a 4 day itinerary to Rio de Janeiro in Brazil for our family."}
  ]
) as stream:
    for text in stream.text_stream:
      print(text, end="", flush=True)
