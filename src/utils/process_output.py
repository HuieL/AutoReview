import re

def extract_domains(output):
    if isinstance(output, list):
        output = ' '.join(output)
    
    domain_pattern = r'Domain \d+: (.+?)\n'
    domains = re.findall(domain_pattern, output)
    return domains

def extract_aspects(output):
    if isinstance(output, list):
        output = ' '.join(output)

    output = re.sub(r'<format>\s*|\s*<format>', '', output).strip()
    aspects = re.split(r'\nAspect \d+: ', output)[1:]
    return aspects

def extract_merged_aspects(output):
    if isinstance(output, list):
        output = ' '.join(output)

    output = re.sub(r'<format>\s*|\s*<format>', '', output).strip()
    aspects = re.split(r'\nMerged aspect \d+: ', output)[1:]
    return aspects

def extract_txt_file(file_name):
    f = open(file_name)
    text = f.read()
    return text
