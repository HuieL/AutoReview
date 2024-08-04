from typing import List, Dict, Any
import tiktoken
from src.model.prompt import FINDING_AREA_PROMPT, ASPECT_PROMPT, ASPECT_WRITING_PROMPT


def area_finding_prompt(paper: Dict[str, Any], domain_number: int = 5) -> str:
    paper_content = paper['content'] if paper['content'] else paper['abstract']
    if not paper_content:
        raise ValueError("Paper has no content or abstract.")
    
    prompt = FINDING_AREA_PROMPT.replace('[PAPER]', paper_content)
    prompt = prompt.replace('[Domain Number]', str(domain_number))
    return prompt

def topic_finding_prompt(paper: Dict[str, Any], domain: str) -> str:
    paper_content = paper['content'] if paper['content'] else paper['abstract']
    if not paper_content:
        raise ValueError("Paper has no content or abstract.")
    
    prompt = ASPECT_PROMPT.replace('[PAPER]', paper_content)
    prompt = prompt.replace('[Domain]', str(domain))
    return prompt

def aspect_writing_prompt(paper: Dict[str, Any], domain: str, aspect: str) -> str:
    paper_content = paper['content'] if paper['content'] else paper['abstract']
    if not paper_content:
        raise ValueError("Paper has no content or abstract.")
    
    prompt = ASPECT_WRITING_PROMPT.replace('[PAPER]', paper_content)
    prompt = prompt.replace('[Domain]', str(domain))
    prompt = prompt.replace('[ASPECT]', str(aspect))
    return prompt


class tokenCounter():
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.model_price = {}
        
    def num_tokens_from_string(self, string:str) -> int:
        return len(self.encoding.encode(string))

    def num_tokens_from_list_string(self, list_of_string:List[str]) -> int:
        num = 0
        for s in list_of_string:
            num += len(self.encoding.encode(s))
        return num
    
    def compute_price(self, input_tokens, output_tokens, model):
        return (input_tokens/1000) * self.model_price[model][0] + (output_tokens/1000) * self.model_price[model][1]

    def text_truncation(self,text, max_len = 1000):
        encoded_id = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(encoded_id[:min(max_len,len(encoded_id))])
