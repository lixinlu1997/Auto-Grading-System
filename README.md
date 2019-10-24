# Project Title

This is an auto grading system designed for Shanghai High School Physics Department.

## Getting Started

To use this system, you have to put boxes inside the test paper for this system to recognize. The boxes have been included in the /Box file and you can add them to your test paper file. The sample template should look like:

![image](Images/sample_template.png)

Then you can put the students' test paper in /original directory, you can merge students in a class in a pdf file. Later the system can do statistic analysis based on classes and whole grade.


### Prerequisites

What things you need to install the software and how to install them

```
pytorch
opencv
PIL
pdf2image
```


## Running the tests

```
python table_detect.py -p original
```
- Then there will be several .csv file in the /stats directory, depending on how many files you have in /original directory. There has been a sample file in directory.

- To do statistic analysis, run
```
python statistic.py   
```
- The class analysis results will be stored in /stats/class directory and grade analysis result will be stored in /stats/grade directory



