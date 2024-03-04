import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
  model="claude-2.1",
  max_tokens=1024,
  messages=[
    {"role": "user", "content": "Give a 4 day itinerary to Tokyo, Japan."}
  ]
)

print(message.content[0].text)
