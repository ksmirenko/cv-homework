# Test results

Note: the same data set was used in all test cases.

## Case 1

Used `opencv_traincascade` with parameters:

* -numStages 10
* -minhitrate 0.999
* -maxFalseAlarmRate 0.5
* -numPos 600
* -numNeg 800

| Measure         | Training data | Test data |
|-----------------|---------------|-----------|
| Total tests     | 1600          | 400       |
| Accuracy        | 95.750%       | 97.000%   |
| Precision       | 97.409%       | 98.958%   |
| Recall          | 94.000%       | 95.000%   |
| True positives  | 752           | 190       |
| True negatives  | 780           | 198       |
| False positives | 20            | 2         |
| False negatives | 48            | 10        |


## Case 2

Used `opencv_traincascade` with parameters:

* -numStages 13
* -minhitrate 0.999
* -maxFalseAlarmRate 0.5
* -numPos 640
* -numNeg 800

| Measure         | Training data | Test data |
|-----------------|---------------|-----------|
| Total tests     | 1600          | 400       |
| Accuracy        | 93.250%       | 94.500%   |
| Precision       | 100.000%      | 100.000%  |
| Recall          | 86.500%       | 89.000%   |
| True positives  | 692           | 178       |
| True negatives  | 800           | 200       |
| False positives | 0             | 0         |
| False negatives | 108           | 22        |


## Case 3

Used `opencv_traincascade` with parameters:

* -numStages 11
* -minhitrate 0.999
* -maxFalseAlarmRate 0.5
* -numPos 620
* -numNeg 800

| Measure         | Training data | Test data |
|-----------------|---------------|-----------|
| Total tests     | 1600          | 400       |
| Accuracy        | 96.812%       | 96.500%   |
| Precision       | 99.083%       | 99.468%   |
| Recall          | 94.500%       | 93.500%   |
| True positives  | 756           | 187       |
| True negatives  | 793           | 199       |
| False positives | 7             | 1         |
| False negatives | 44            | 13        |

Note: the cascade was trained with no previously provided data.

