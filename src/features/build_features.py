import sys
from pathlib import Path









# Steps Followed Here : 
# 1. Remove Customer ID from all the extracts
# 2. Drop Null Values (Only 1 null row missing so far)
# 3. Drop Duplicate (Only 1 values seen so far)
# 4. Convert All the Numerical Columns to int except Total Spend


def main():
    print("hello")


if __name__ == "main":
    for ind in range(1,4):
        input_file_name = sys.argv[ind]
        curr_path = Path(__file__)
        root_path = curr_path.parent.parent.parent
        


