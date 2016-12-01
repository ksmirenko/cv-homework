# Test results

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
| True positives  | 752           | 19        |
| True negatives  | 780           | 19        |
| False positives | 20            | 2         |
| False negatives | 48            | 10        |
