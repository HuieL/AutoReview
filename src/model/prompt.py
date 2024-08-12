
FINDING_AREA_PROMPT = '''
Here is the content from a research paper:
---
[PAPER]
---

<instruction>
DOMAIN CANDIDATES: Artificial Intelligence (AI), Computation and Language (CL), Computational Complexity (CC), Computational Engineering, Finance, and Science (CE), Computational Geometry (CG), Computer Science and Game Theory (GT), Computer Vision and Pattern Recognition (CV), Computers and Society (CY), Cryptography and Security (CR), Data Structures and Algorithms (DS), Databases (DB), Digital Libraries (DL), Discrete Mathematics (DM), Distributed, Parallel, and Cluster Computing (DC), Emerging Technologies (ET), Formal Languages and Automata Theory (FL), General Literature (GL), Graphics (GR), Hardware Architecture (AR), Human-Computer Interaction (HC), Information Retrieval (IR), Information Theory (IT), Machine Learning (LG), Logic in Computer Science (LO), Mathematical Software (MS), Multiagent Systems (MA), Multimedia (MM), Networking and Internet Architecture (NI), Neural and Evolutionary Computing (NE), Numerical Analysis (NA), Operating Systems (OS), Other Computer Science (OH), Performance (PF), Programming Languages (PL), Robotics (RO), Social and Information Networks (SI), Software Engineering (SE), Sound (SD), Symbolic Computation (SC), Systems and Control (SY).
---
The number of output domains: [Domain Number]
---
According to the DOMAIN CANDIDATES and paper info given, identify and output the most related domain(s) of this paper.
Return in the format:
<format>
Domain 1: [Domain 1 Name]

...

Domain [Domain Number]: [Domain [Domain Number] Name]
<format>
Output:
'''

ASPECT_PROMPT = '''
You are an expert in the domain of [DOMAIN] who wants to review a paper in this domain and write a comprehensive comments for it.
You are provided the content of the research paper:
---
[PAPER]
---

<instruction>
Each aspect follows with a brief sentence to describe what this aspect focuses on.
You need to generate a outline based on the paper content to make the outline show comprehensive insights of what aspects this paper should be reviewed from.
Return the in the format:
<format>
Aspect 1: [Aspect Description]

...

Aspect K: [Aspect Description]
<format>
Output:
'''

ASPECT_WRITING_PROMPT = '''
You are an expert reviewer in the field of [DOMAIN]. 
You have been asked to provide a detailed evaluation of a specific aspect of a research paper. 
Here is the content of the research paper:
---
[PAPER]
---
The aspect you need to focus on is:
Aspect: [ASPECT]
---

<instruction>
Based on the given aspect and the paper content, provide a comprehensive and critical evaluation. Your comments should:
1. Assess the paper's strengths and weaknesses related to this aspect.
2. Provide specific examples from the paper to support your assessment.
3. Offer constructive feedback and suggestions for improvement where applicable.
4. Consider the relevance and impact of this aspect on the overall quality of the research.
5. If appropriate, compare the paper's approach in this aspect to current standards or state-of-the-art in the field.
Your evaluation should be thorough, balanced, and insightful, reflecting your expertise in [DOMAIN]. Aim for a response of approximately 200-300 words.
Return your evaluation in the following format:
<evaluation>
[Your detailed comments and analysis here]
</evaluation>
</instruction>
Output:
'''

MERGING_ASPECTS_PROMPT = '''
You are an expert reviewer in the field of [DOMAIN]. 
You have been asked to provide a detailed evaluation of a specific aspect of a research paper. 
Here is the content of the research paper:
---
[PAPER]
---
The list of relevant aspects that are already given for this paper are as follows:
---
Aspect 1: [Aspect Description]

...

Aspect K: [Aspect Description]
---

<instruction>
Based on these given aspects and the paper content, determine whether the aspects and their desciptions are similar enough with each other, and merge these two aspects together by:
1. Assess the similarities between the two descriptions and the aspect title.
2. If similar enough, return the merged description and rename the two merged aspects with the format.
Your evaluation should be thorough, balanced, and insightful, reflecting your expertise in [DOMAIN]. Aim for a response of approximately 200-300 words.
Return your evaluation in the following format:
<evaluation>
If merged: [ASPECT_1]/[ASPECT_2]: [Aspect description]
Else: Leave the aspect as it was before ([ASPECT]: [Aspect description])
</evaluation>
</instruction>
Output:
'''
