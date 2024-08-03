from torch_geometric.data import Data, Dataset

class CitationGraph(Dataset):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data
        self.num_papers = len(data.arxiv_id)

    def len(self):
        return self.num_papers

    def get(self, idx):
        return {
            'arxiv_id': self.data.arxiv_id[idx],
            'title': self.data.title[idx],
            'content': self.data.content[idx],
            'abstract': self.data.abstract[idx],
            'comment': self.data.comment[idx],
            'decision': self.data.decision[idx] if hasattr(self.data, 'decision') else None
        }
    
