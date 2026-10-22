## Overall

| system | n | EM | F1 | Cover | recall_all | recall_hybrid | recall_wikipedia | recall_graph | conflict_rate | retry_rate | EM_when_retried | avg_confidence | avg_llm_calls | avg_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev_closed_book | 100 | 28.0 | 31.3 | 28.0 | 0.0 |  |  |  |  |  |  |  | 1.0 | 0.7 |
| dev_single_rag | 100 | 39.0 | 43.4 | 43.0 | 70.0 | 70.0 |  |  |  |  |  |  | 1.0 | 7.8 |
| dev_full | 100 | 53.0 | 57.8 | 59.0 | 96.5 | 89.2 | 75.0 | 92.5 | 9.0 | 9.0 | 55.6 | 0.7 | 4.2 | 39.8 |
| dev_full_no_wikipedia | 100 | 45.0 | 52.7 | 51.0 | 95.2 | 87.8 |  | 91.5 | 3.0 | 3.0 | 0.0 | 0.8 | 4.1 | 29.9 |
| dev_full_no_graph | 100 | 41.0 | 46.9 | 47.0 | 90.8 | 88.8 | 75.2 |  | 7.0 | 7.0 | 42.9 | 0.8 | 4.2 | 32.1 |
| dev_full_no_hybrid | 100 | 47.0 | 52.5 | 52.0 | 93.2 |  | 73.8 | 92.0 | 9.0 | 9.0 | 55.6 | 0.8 | 4.2 | 31.5 |
| dev_full_no_conflict | 100 | 53.0 | 57.8 | 59.0 | 96.5 | 89.2 | 75.2 | 92.5 | 9.0 | 0.0 |  | 0.7 | 4.0 | 33.7 |

## By question type

| system | type | n | EM | F1 | Cover |
|---|---|---|---|---|---|
| dev_closed_book | bridge_comparison | 24 | 45.8 | 48.3 | 45.8 |
| dev_closed_book | comparison | 24 | 70.8 | 70.8 | 70.8 |
| dev_closed_book | compositional | 43 | 0.0 | 3.1 | 0.0 |
| dev_closed_book | inference | 9 | 0.0 | 15.3 | 0.0 |
| dev_single_rag | bridge_comparison | 24 | 58.3 | 60.8 | 58.3 |
| dev_single_rag | comparison | 24 | 75.0 | 75.1 | 79.2 |
| dev_single_rag | compositional | 43 | 11.6 | 16.7 | 18.6 |
| dev_single_rag | inference | 9 | 22.2 | 39.6 | 22.2 |
| dev_full | bridge_comparison | 24 | 66.7 | 66.7 | 66.7 |
| dev_full | comparison | 24 | 87.5 | 87.5 | 87.5 |
| dev_full | compositional | 43 | 34.9 | 41.7 | 48.8 |
| dev_full | inference | 9 | 11.1 | 31.8 | 11.1 |
| dev_full_no_wikipedia | bridge_comparison | 24 | 50.0 | 52.4 | 50.0 |
| dev_full_no_wikipedia | comparison | 24 | 79.2 | 79.2 | 79.2 |
| dev_full_no_wikipedia | compositional | 43 | 27.9 | 39.8 | 41.9 |
| dev_full_no_wikipedia | inference | 9 | 22.2 | 44.8 | 22.2 |
| dev_full_no_graph | bridge_comparison | 24 | 37.5 | 37.5 | 37.5 |
| dev_full_no_graph | comparison | 24 | 83.3 | 83.3 | 83.3 |
| dev_full_no_graph | compositional | 43 | 25.6 | 36.1 | 39.5 |
| dev_full_no_graph | inference | 9 | 11.1 | 26.6 | 11.1 |
| dev_full_no_hybrid | bridge_comparison | 24 | 54.2 | 54.2 | 54.2 |
| dev_full_no_hybrid | comparison | 24 | 79.2 | 79.2 | 79.2 |
| dev_full_no_hybrid | compositional | 43 | 34.9 | 43.3 | 46.5 |
| dev_full_no_hybrid | inference | 9 | 0.0 | 21.2 | 0.0 |
| dev_full_no_conflict | bridge_comparison | 24 | 66.7 | 66.7 | 66.7 |
| dev_full_no_conflict | comparison | 24 | 87.5 | 87.5 | 87.5 |
| dev_full_no_conflict | compositional | 43 | 34.9 | 41.7 | 48.8 |
| dev_full_no_conflict | inference | 9 | 11.1 | 31.8 | 11.1 |

### Confidence calibration: dev_full

| confidence | n | EM | F1 |
|---|---|---|---|
| [0.0, 0.4] | 17 | 17.6 | 23.2 |
| [0.4, 0.6] | 6 | 50.0 | 62.5 |
| [0.6, 0.8] | 12 | 33.3 | 35.7 |
| [0.8, 1.0] | 65 | 66.2 | 70.5 |

### Confidence calibration: dev_full_no_wikipedia

| confidence | n | EM | F1 |
|---|---|---|---|
| [0.0, 0.4] | 16 | 12.5 | 20.9 |
| [0.4, 0.6] | 5 | 60.0 | 66.7 |
| [0.6, 0.8] | 19 | 52.6 | 55.3 |
| [0.8, 1.0] | 60 | 50.0 | 59.3 |

### Confidence calibration: dev_full_no_graph

| confidence | n | EM | F1 |
|---|---|---|---|
| [0.0, 0.4] | 18 | 11.1 | 15.7 |
| [0.4, 0.6] | 2 | 100.0 | 100.0 |
| [0.6, 0.8] | 26 | 42.3 | 50.7 |
| [0.8, 1.0] | 54 | 48.1 | 53.5 |

### Confidence calibration: dev_full_no_hybrid

| confidence | n | EM | F1 |
|---|---|---|---|
| [0.0, 0.4] | 17 | 11.8 | 17.2 |
| [0.4, 0.6] | 6 | 50.0 | 54.2 |
| [0.6, 0.8] | 21 | 42.9 | 46.9 |
| [0.8, 1.0] | 56 | 58.9 | 65.2 |

### Confidence calibration: dev_full_no_conflict

| confidence | n | EM | F1 |
|---|---|---|---|
| [0.0, 0.4] | 16 | 12.5 | 18.4 |
| [0.4, 0.6] | 8 | 50.0 | 59.4 |
| [0.6, 0.8] | 11 | 36.4 | 39.0 |
| [0.8, 1.0] | 65 | 66.2 | 70.5 |