import datetime

LOG_FILE = "./chat_log.txt"

def log_interaction(user_input, response_text, token_counter):
    timestamp = datetime.datetime.now().isoformat()
    token_counts = {
        'total_embedding_token_count': token_counter.total_embedding_token_count,
        'prompt_llm_token_count': token_counter.prompt_llm_token_count,
        'completion_llm_token_count': token_counter.completion_llm_token_count,
        'total_llm_token_count': token_counter.total_llm_token_count,
        'prompt': token_counter.llm_token_counts[0].prompt,
        'completion': token_counter.llm_token_counts[0].completion
    }
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp}: User: {user_input}\n")
        f.write(f"{timestamp}: Bot: {response_text}\n")
        f.write("Embedding Tokens: {}\n".format(token_counts['total_embedding_token_count']))
        f.write("LLM Prompt Tokens: {}\n".format(token_counts['prompt_llm_token_count']))
        f.write("LLM Completion Tokens: {}\n".format(token_counts['completion_llm_token_count']))
        f.write("Total LLM Token Count: {}\n".format(token_counts['total_llm_token_count']))
        f.write("Prompt: {}\n".format(token_counts['prompt']))
        f.write("completion: {}\n".format(token_counts['completion']))
        f.write("=======================================================\n\n\n")
        
