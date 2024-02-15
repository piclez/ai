require 'net/http'
require 'uri'
require 'json'

uri = URI('http://localhost:11434/api/chat')

request = Net::HTTP::Post.new(uri)
request.content_type = 'application/json'
request.body = JSON.dump({
 model: 'mistral',
 messages: [
   {
     role: 'user',
     content: 'How can I covert a PDF into text?',
   }
 ],
 stream: false
})

response = Net::HTTP.start(uri.hostname, uri.port) do |http|
 http.read_timeout = 120
 http.request(request)
end

puts response.body
