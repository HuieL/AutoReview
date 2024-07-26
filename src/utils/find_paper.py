import requests
from bs4 import BeautifulSoup
import re
import tarfile
import os
import xml.etree.ElementTree as ET
import urllib.parse
import tempfile
from fuzzywuzzy import fuzz
import time
import torch
from torch_geometric.data import Data
import ipdb
from tqdm import tqdm
from requests.exceptions import RequestException, ProxyError
import gzip
import backoff


@backoff.on_exception(backoff.expo, RequestException, max_tries=2)
def download_arxiv_source(arxiv_id):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        return temp_file.name

def process_main_tex(main_tex_content, base_dir):
    def replace_input(match):
        input_file = match.group(1)
        if not input_file.endswith('.tex'):
            input_file += '.tex'
        input_path = os.path.join(base_dir, input_file)
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                return f.read()
        return match.group(0)  # If file not found, keep original \input command

    # Replace all \input commands with their file contents
    processed_content = re.sub(r'\\input{([^}]+)}', replace_input, main_tex_content)
    return processed_content

def extract_tex_file(tar_gz_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with tarfile.open(tar_gz_path, "r:gz") as tar:
                        tar.extractall(path=temp_dir)
                except tarfile.ReadError:
                    # If it's not a gzip file, try opening it as a regular tar file
                    with tarfile.open(tar_gz_path, "r:") as tar:
                        tar.extractall(path=temp_dir)
                
                main_tex_file = find_main_tex_file(temp_dir)
                if main_tex_file:
                    with open(main_tex_file, 'r', encoding='utf-8', errors='ignore') as tex_file:
                        main_tex_content = tex_file.read()
                    return process_input_commands(main_tex_content, os.path.dirname(main_tex_file))
                else:
                    print(f"No main .tex file found in {tar_gz_path}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Failed to extract tar file after {max_retries} attempts: {e}")
                return None

def find_main_tex_file(directory):
    tex_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
    
    if not tex_files:
        return None
    
    # Check for a file named 'main.tex' or similar
    main_file_candidates = [f for f in tex_files if re.match(r'(main|paper|article)\.tex', os.path.basename(f), re.IGNORECASE)]
    if main_file_candidates:
        return main_file_candidates[0]
    
    # If no obvious main file, check for \documentclass and \begin{document}
    for file in tex_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            if '\\documentclass' in content and '\\begin{document}' in content:
                return file
    
    # If still not found, return the largest .tex file
    return max(tex_files, key=os.path.getsize)

def remove_comments(tex_content):
    lines = tex_content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line.startswith('%') or stripped_line.startswith('\\%'):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def extract_main_text(tex_content):
    pattern = re.compile(r'\\begin{document}(.*?)\\end{document}', re.DOTALL)
    match = pattern.search(tex_content)
    if match:
        main_text = match.group(1).strip()
        return remove_comments(main_text)
    return None

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_arxiv_main_text(arxiv_id):
    try:
        tar_gz_path = download_arxiv_source(arxiv_id)
        try:
            tex_content = extract_tex_file(tar_gz_path)
            if tex_content:
                main_text = extract_main_text(tex_content)
                return main_text
        finally:
            if os.path.exists(tar_gz_path):
                os.unlink(tar_gz_path)  # Delete the temporary tar.gz file
    except Exception as e:
        print(f"Error extracting main text for {arxiv_id}: {str(e)}")
    return None

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_arxiv_abstract(arxiv_id):
    try:
        base_url = "http://export.arxiv.org/api/query?"
        query = f"id_list={arxiv_id}"

        response = requests.get(base_url + query)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', namespace)
        if entry is None:
            return None

        summary = entry.find('atom:summary', namespace)
        if summary is not None:
            return summary.text.strip()
    except Exception as e:
        print(f"Error fetching abstract for {arxiv_id}: {str(e)}")
    return None

def get_arxiv_id_from_url(paper_url):
    _, arxiv_url = extract_paper_info(paper_url)
    return arxiv_url.split('/')[-1]

############### Scripts to get references

def process_input_commands(content, base_dir):
    def replace_input(match):
        input_file = match.group(1)
        if not input_file.endswith('.tex'):
            input_file += '.tex'
        input_path = os.path.join(base_dir, input_file)
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                return process_input_commands(f.read(), os.path.dirname(input_path))
        return ''  # If file not found, replace with empty string

    # Replace all \input commands with their file contents
    processed_content = re.sub(r'\\input\{([^}]+)\}', replace_input, content)
    return processed_content

def find_reference_files(tex_content):
    # Look for \bibliography{...}, \addbibresource{...}, or \bibdata{...} commands
    ref_commands = re.findall(r'\\(?:bibliography|addbibresource|bibdata){([^}]+)\}', tex_content)
    ref_files = []
    for command in ref_commands:
        ref_files.extend(command.split(','))
    return [f"{ref.strip()}" for ref in ref_files]

def parse_bib_content(bib_content):
    # Split the content into entries, ignoring non-entry text
    entries = re.split(r'(@\w+\s*\{[^@]*?\n\})', bib_content, flags=re.DOTALL)
    titles = []
    
    for entry in entries:
        if entry.strip().startswith('@'):
            # Extract the title from valid entries
            title_match = re.search(r'title\s*=\s*["{](.+?)["}]', entry, re.DOTALL | re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                # Remove any newlines and extra spaces from the title
                title = re.sub(r'\s+', ' ', title)
                titles.append(title)
    
    return titles

def parse_bbl_content(bbl_content):
    entries = re.split(r'\\bibitem', bbl_content)[1:]  # Split by \bibitem and remove the first empty element
    titles = []

    for entry in entries:
        # Remove any newline characters and extra spaces
        entry = re.sub(r'\s+', ' ', entry).strip()
        
        # Try to find the title
        title = None
        
        # Look for patterns that typically indicate a title
        title_patterns = [
            r'\\newblock\s*(.*?)\s*\\newblock',  # Captures content between first two \newblock commands
            r'{\\em\s+(.*?)}',  # Captures content within {\em ...}
            r'``(.*?)\'\'',  # Captures content between double quotes
            r'"(.*?)"',  # Captures content between smart quotes
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, entry)
            if match:
                title = match.group(1).strip()
                # Remove any remaining LaTeX commands
                title = re.sub(r'\\[a-zA-Z]+(\[.*?\])?({.*?})?', '', title)
                break
        
        if title:
            titles.append(title)
        else:
            # If no title found, add a placeholder
            titles.append("Title not found")

    return titles

def extract_reference_titles(arxiv_id, main_text):
    cited_titles = []
    try:
        tar_file = download_arxiv_source(arxiv_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(path=temp_dir)
            except tarfile.ReadError:
                # If it's not a gzip file, try opening it as a regular tar file
                with tarfile.open(tar_file, 'r:') as tar:
                    tar.extractall(path=temp_dir)
            
            # Find the main .tex file
            main_tex_file = find_main_tex_file(temp_dir)
            
            if not main_tex_file:
                print(f"Could not find main .tex file for {arxiv_id}")
                return cited_titles

            # Process the main .tex file and all its inputs
            with open(main_tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = process_input_commands(f.read(), os.path.dirname(main_tex_file))
            
            ref_files = find_reference_files(tex_content)
            if not ref_files:
                print(f"No reference files found in the processed .tex content for {arxiv_id}")
                return cited_titles

            # Extract titles from each referenced file
            for ref_file in ref_files:
                ref_path = os.path.join(os.path.dirname(main_tex_file), ref_file)
                if not os.path.exists(ref_path):
                    # Try adding extensions if the file doesn't exist
                    for ext in ['.bib', '.bbl']:
                        test_path = ref_path + ext
                        if os.path.exists(test_path):
                            ref_path = test_path
                            break
                
                if os.path.exists(ref_path):
                    try:
                        with open(ref_path, 'r', encoding='utf-8', errors='ignore') as ref_file:
                            content = ref_file.read()
                            if ref_path.endswith('.bib'):
                                titles = parse_bib_content(content)
                            elif ref_path.endswith('.bbl'):
                                titles = parse_bbl_content(content)
                            else:
                                print(f"Unrecognized reference file type: {ref_path}")
                                continue
                            cited_titles.extend(titles)
                    except UnicodeDecodeError:
                        print(f"Could not read file {ref_path} as UTF-8")
                else:
                    print(f"Referenced file not found: {ref_file}")

    except Exception as e:
        print(f"An error occurred while processing {arxiv_id}: {str(e)}")
    finally:
        if 'tar_file' in locals() and os.path.exists(tar_file):
            os.unlink(tar_file)  # Delete the temporary tar.gz file

    return cited_titles

############### Utils to extract info of cited papers

def clean_title(title):
    # Remove special characters and extra spaces
    return re.sub(r'\W+', ' ', title).strip().lower()

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_arxiv_id(title, max_results=10, similarity_threshold=80):
    cleaned_title = clean_title(title)
    words = cleaned_title.split()
    
    query = '+AND+'.join(f'all:{urllib.parse.quote(word)}' for word in words)
    
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f'{base_url}search_query={query}&start=0&max_results={max_results}'
    
    response = requests.get(search_query)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    
    best_match = None
    highest_similarity = 0
    
    for entry in root.findall('atom:entry', namespace):
        entry_title = entry.find('atom:title', namespace).text
        similarity = fuzz.ratio(cleaned_title, clean_title(entry_title))
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = entry
        
        if similarity == 100:  # Exact match found
            break
    
    if best_match is not None and highest_similarity >= similarity_threshold:
        arxiv_url = best_match.find('atom:id', namespace).text
        arxiv_id = arxiv_url.split('/')[-1]
        return arxiv_id
    else:
        return None
    
def get_arxiv_ids(titles):
    arxiv_ids = []
    print("Processing id of references")
    for title in tqdm(titles, desc="Searching for references", unit="title"):
        try:
            arxiv_id = get_arxiv_id(title)
            arxiv_ids.append(arxiv_id)
            time.sleep(1)  # Respect arXiv's rate limit, increased to 3 seconds
        except Exception as e:
            arxiv_ids.append(None)
    return arxiv_ids

############### Utils to build up graphs and record text info of these papers
def construct_citation_graph(root_arxiv_id, root_title, cited_titles, cited_arxiv_ids):
    # Filter out None values from cited_arxiv_ids and keep track of valid indices
    valid_cited = [(i, id) for i, id in enumerate(cited_arxiv_ids) if id is not None]
    valid_indices, valid_cited_ids = zip(*valid_cited) if valid_cited else ([], [])

    # Create a list of all valid arXiv IDs (including root) and their corresponding titles
    all_arxiv_ids = [root_arxiv_id] + list(valid_cited_ids)
    print(f"Found cited papers on arxiv: {len(all_arxiv_ids)}/{len(cited_titles)+1}")
    all_titles = [root_title] + [cited_titles[i] for i in valid_indices]

    # Create a mapping from arXiv ID to node index
    id_to_index = {id: i for i, id in enumerate(all_arxiv_ids)}

    # Create node features (optional, you can modify this as needed)
    num_nodes = len(all_arxiv_ids)
    node_features = torch.eye(num_nodes)  # One-hot encoding for each node

    # Create edge index
    edge_index = [[0, id_to_index[id]] for id in valid_cited_ids]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the PyG Data object
    data = Data(x=node_features, edge_index=edge_index)

    # Add arXiv IDs and titles as additional attributes
    data.arxiv_ids = all_arxiv_ids
    data.titles = all_titles

    # Add a mapping from original citation index to new graph index
    data.citation_to_graph_index = {i: id_to_index[id] for i, id in zip(valid_indices, valid_cited_ids)}

    return data, valid_cited_ids

def merge_citation_graphs(graphs):
    # Collect all unique arXiv IDs and titles
    all_arxiv_ids = []
    all_titles = []
    for graph in graphs:
        for arxiv_id, title in zip(graph.arxiv_ids, graph.titles):
            if arxiv_id not in all_arxiv_ids:
                all_arxiv_ids.append(arxiv_id)
                all_titles.append(title)

    # Create a mapping from arXiv ID to new node index
    id_to_index = {id: i for i, id in enumerate(all_arxiv_ids)}

    # Create new node features
    num_nodes = len(all_arxiv_ids)
    node_features = torch.eye(num_nodes)

    # Create new edge index
    edge_list = []
    for graph in graphs:
        for edge in graph.edge_index.t().tolist():
            source_id = graph.arxiv_ids[edge[0]]
            target_id = graph.arxiv_ids[edge[1]]
            new_source = id_to_index[source_id]
            new_target = id_to_index[target_id]
            edge_list.append([new_source, new_target])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create the merged PyG Data object
    merged_data = Data(x=node_features, edge_index=edge_index)
    merged_data.arxiv_ids = all_arxiv_ids
    merged_data.titles = all_titles

    return merged_data

def update_graph_with_content(graph):
    # Create masks for valid content and abstracts
    content_mask = [content is not None for content in graph.content]
    abstract_mask = [abstract is not None for abstract in graph.abstract]
    
    # Combine masks
    valid_mask = [c and a for c, a in zip(content_mask, abstract_mask)]
    
    # Update edge index
    new_edge_index = []
    for edge in graph.edge_index.t():
        if valid_mask[edge[0]] and valid_mask[edge[1]]:
            new_source = sum(valid_mask[:edge[0]])
            new_target = sum(valid_mask[:edge[1]])
            new_edge_index.append([new_source, new_target])
    
    if not new_edge_index:
        print("No valid edges remaining after filtering.")
        return None
    
    graph.edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
    
    # Update other attributes
    graph.arxiv_ids = [id for id, valid in zip(graph.arxiv_ids, valid_mask) if valid]
    graph.titles = [title for title, valid in zip(graph.titles, valid_mask) if valid]
    graph.content = [content for content in graph.content if content is not None]
    graph.abstract = [abstract for abstract in graph.abstract if abstract is not None]
    
    return graph

def build_citation_tree(root_arxiv_id, paper_title):
    try:
        main_text = get_arxiv_main_text(root_arxiv_id)
        abstract = get_arxiv_abstract(root_arxiv_id)

        cited_titles = extract_reference_titles(root_arxiv_id, main_text or '')
        cited_arxiv_ids = get_arxiv_ids(cited_titles)

        graph, valid_cited_ids = construct_citation_graph(root_arxiv_id, paper_title, cited_titles, cited_arxiv_ids)

        graph.content = [main_text]
        graph.abstract = [abstract]
        graph.title = [paper_title]

        print("Processing contents of cited papers")
        for id, title in tqdm(zip(valid_cited_ids, [cited_titles[i] for i in graph.citation_to_graph_index.keys()]), 
                              desc=f"Extracting content of citations"):
            try:
                content = get_arxiv_main_text(id)
                abstract = get_arxiv_abstract(id)
                graph.content.append(content)
                graph.abstract.append(abstract)
                graph.title.append(title)
            except Exception as e:
                graph.content.append(None)
                graph.abstract.append(None)
                graph.title.append(title)

        return update_graph_with_content(graph)
    except Exception as e:
        print(f"Error building citation tree for {paper_title}: {str(e)}")
        return None

### Example Usage:
# arxiv_id = get_arxiv_id(title)
# citation_tree = build_citation_tree(root_arxiv_id, paper_title)
