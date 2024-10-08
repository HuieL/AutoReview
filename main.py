import sys
import os

path = os.getcwd()
sys.path.append(os.path.abspath(path + "/src/utils"))
from src.utils.find_paper import *

def main():
    base_folder = path + "/dataset"
    output_folder = path + "/processed"
    process_conferences(base_folder, output_folder)
    return 0


if __name__ == "__main__":
    main()