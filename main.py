import sys
import os

path = os.getcwd()
sys.path.append(os.path.abspath(path + "/src/utils"))
from find_paper import *

def main():
    base_folder = path + "/dataset"
    output_folder = path + "/processed"
    process_conferences(base_folder, output_folder)
    print(torch.load(path + "/processed/ICLR_2017.pt"))
    return 0


if __name__ == "__main__":
    main()