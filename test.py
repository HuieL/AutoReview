import torch
from src.utils.prompting import area_finding_prompt, topic_finding_prompt
from src.utils.load_data import CitationGraph
from src.model.llm import APIModel
from typing import Dict, Any
import random


random.seed(42)

def select_random_paper_with_comment(dataset: CitationGraph) -> Dict[str, Any]:
    papers_with_comments = [i for i in range(len(dataset)) if dataset[i]['comment']]
    if not papers_with_comments:
        raise ValueError("No papers with comments found in the database.")
    random_index = random.choice(papers_with_comments)
    return dataset[random_index]


# Example usage
Counter = tokenCounter()
model  = APIModel(model = "gpt-4o-2024-05-13", api_key = "YOUR_KEYS", api_url = "https://api.openai.com/v1/chat/completions")
db = torch.load("./dataset/ICLR_2018.pt")
paper_dataset = CitationGraph(db)

random_paper = select_random_paper_with_comment(paper_dataset)
area_finding_prompt = area_finding_prompt(random_paper)

prompts = [area_finding_prompt]
outputs = model.batch_chat(text_batch=prompts, temperature=1)
domains = extract_domains(outputs)

print(domains)

# For instance, we select the first and the third specialized domains to output detailed aspects ...
prompts = [topic_finding_prompt(random_paper, domains[0]), topic_finding_prompt(random_paper, domains[2])]
outputs = model.batch_chat(text_batch=prompts, temperature=1)

with open('extracted_aspects.txt', 'w', encoding='utf-8') as f:
    f.write(f"\n\n{'-'*200}\n\n".join(outputs))

# You can use these test codes in a notebook to quickly start...
# You can check the aspects in the .txt file extracted...
# Then, aspect_writing_prompt() to merge these aspects from different domain perspective...
# You need to check the output and adjust the prompts ...

