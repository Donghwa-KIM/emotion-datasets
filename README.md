# Datasets for emotion classification using audio and text
Popular emotion-datasets

- We give the most popular emotion datasets as rawdata format to use audio array or text string
- The raw dataset was collected from public sites
    - https://github.com/A2Zadeh/CMU-MultimodalSDK
    - https://sail.usc.edu/iemocap/
    - https://github.com/declare-lab/MELD
- Rearranged raw datasets [[emotion_data.zip]()] 
- And, `src/preprocess.py` easily converts the raw dataset as a dataframe.pkl.
    - `dataframe.pkl`s were saved in [[data.zip]()]
- Non-existent row was removed, for example, negative time interval or empty audio files ...
---

- Implementation

```shell
python ./src/preprocess.py --config_file ./configs/cmu_mosei.yaml
```

- results

```
/data/CMU_mosei_data.pkl
```


|index| wav_id |       start |     end | sentiment |    happy |      sad |    anger | surprise | disgust | fear |    split |  text |                                             audio |                                             audio |
|-------:|-------:|------------:|--------:|----------:|---------:|---------:|---------:|---------:|--------:|-----:|---------:|------:|--------------------------------------------------:|--------------------------------------------------:|
|      0 | --qXJuDtHPw |  23.199 |    30.325 | 1.000000 | 0.666667 | 0.000000 |      0.0 |     0.0 |  0.0 | 0.000000 | valid | I see that a writer is somebody who has an inc... | [0.025634766, 0.03857422, 0.051208496, 0.04162... |
|      1 | -3g5yACwYnA |  82.753 |   100.555 | 1.000000 | 0.666667 | 0.666667 |      0.0 |     0.0 |  0.0 | 0.666667 | train | Key is part of the people that we use to solve... | [0.00033569336, -0.0002746582, 0.0005493164, 0... |
|      2 | -3g5yACwYnA | 119.919 |   125.299 | 0.666667 | 0.000000 | 0.000000 |      0.0 |     0.0 |  0.0 | 0.000000 | train | They've been able to find solutions or at leas... | [-0.0064086914, -0.007507324, -0.0072021484, -... |
|      3 | -3g5yACwYnA |   4.840 |    14.052 | 0.000000 | 0.666667 | 0.666667 |      0.0 |     0.0 |  0.0 | 0.333333 | train | Key Polymer brings a technical aspect to our o... | [9.1552734e-05, 0.0002746582, 0.00088500977, 0... |
|        |             |         |           |          |          |          |          |         |      |          |       |                                                   |                                                   |

- Implementation

```shell
python ./src/preprocess.py --config_file ./configs/iemocap.yaml
```
- results

```
/data/iemocap_data.pkl
```

|               Anger | Disgust | Excited | Fear | Frustration | Happiness | Neutral state | Other | Sadness | Surprise | text |               audio |                                              part |                                             audio |
|--------------------:|--------:|--------:|-----:|------------:|----------:|--------------:|------:|--------:|---------:|-----:|--------------------:|--------------------------------------------------:|--------------------------------------------------:|
| Ses01F_impro01_F000 |     0.0 |     0.0 |  0.0 |         0.0 |       0.0 |           0.0 |   3.0 |     0.0 |      0.0 |  0.0 |          Excuse me. | [-0.0050354004, -0.0049743652, -0.0038146973, ... | [-0.02835083, -0.039916992, -0.04159546, -0.03... |
| Ses01F_impro01_F001 |     0.0 |     0.0 |  0.0 |         0.0 |       0.0 |           0.0 |   3.0 |     0.0 |      0.0 |  0.0 |               Yeah. | [0.0009460449, -0.0009460449, -0.0007019043, -... | [0.005004883, 0.007507324, 0.009735107, 0.0095... |
| Ses01F_impro01_F002 |     0.0 |     0.0 |  0.0 |         0.0 |       0.0 |           0.0 |   2.0 |     0.0 |      0.0 |  1.0 | Is there a problem? | [-0.00036621094, -0.00015258789, 0.0004272461,... | [3.0517578e-05, 0.000579834, 0.0013122559, 0.0... |
| Ses01F_impro01_F003 |     1.0 |     0.0 |  0.0 |         0.0 |       1.0 |           0.0 |   1.0 |     0.0 |      0.0 |  0.0 |            You did. | [-0.0048828125, -0.0046691895, -0.005279541, -... | [-0.0022583008, -0.0010375977, 0.00289917, 0.0... |
|                     |         |         |      |             |           |               |       |         |          |      |                     |                                                   |                                                   |

- Implementation


```shell
python ./src/preprocess.py --config_file ./configs/meld.yaml
```

- results

```
/data/meld_data.pkl
```

|   | sr. no |                                         Utterance |  Speaker |  Emotion | Sentiment | Dialogue_ID | Utterance_ID | Season | Episode |    StartTime |      EndTime | part |                                             audio |
|--:|-------:|--------------------------------------------------:|---------:|---------:|----------:|------------:|-------------:|-------:|--------:|-------------:|-------------:|-----:|--------------------------------------------------:|
| 0 |      1 |    Oh my God, he’s lost it. He’s totally lost it. |   Phoebe |  sadness |  negative |           0 |            0 |      4 |       7 | 00:20:57,256 | 00:21:00,049 |  dev | [-0.02835083, -0.039916992, -0.04159546, -0.03... |
| 1 |      2 |                                             What? |   Monica | surprise |  negative |           0 |            1 |      4 |       7 | 00:21:01,927 | 00:21:03,261 |  dev | [0.005004883, 0.007507324, 0.009735107, 0.0095... |
| 2 |      3 | Or! Or, we could go to the bank, close our acc... |     Ross |  neutral |   neutral |           1 |            0 |      4 |       4 | 00:12:24,660 | 00:12:30,915 |  dev | [3.0517578e-05, 0.000579834, 0.0013122559, 0.0... |
| 3 |      4 |                                  You’re a genius! | Chandler |      joy |  positive |           1 |            1 |      4 |       4 | 00:12:32,334 | 00:12:33,960 |  dev | [-0.0022583008, -0.0010375977, 0.00289917, 0.0... |
|   |        |                                                   |          |          |           |             |              |        |         |              |              |      |                                                   |
