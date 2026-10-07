import Config

if key = System.get_env("ANTHROPIC_API_KEY") do
  config :req_llm, anthropic_api_key: key
end
