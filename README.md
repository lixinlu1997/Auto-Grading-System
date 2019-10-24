This is an auto grading system for Shanghai High School Physics Department

# Requirement

-opencv
-PIL
-pytorch
-pdf2image

# Introduction

-This system can recognize the boxes in a pdf file and grade a test paper according to the solution given in solution.csv
-The syntax for solution.csv has been included in "solution 中文输入说明.csv"

# Usage

-First, put the template file in the root directory. It should be as clear as possible.
-Second, create the solution for this test paper in solution.csv and store in root directory.
-Finally, put the students' test paper in /original file.

-Run
'''bash
python table_detect.py -original
'''

