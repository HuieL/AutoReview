import sys
import os

path = os.getcwd()
sys.path.append(os.path.abspath(path + "/src/utils"))
from find_paper import *
from llm_model import *

def main():
    #base_folder = path + "/dataset/ICLR_2017"
    #output_folder = path + "/dataset/processed"
    #graphs = process_conference(base_folder)
    #save_graph(graphs, output_folder, "ICLR_2017")
    print(torch.load("dataset/processed/ICLR_2017.pt"))
    return 0


if __name__ == "__main__":
    main()