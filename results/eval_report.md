## Overall

| system | n | EM | F1 | Cover | recall_all | recall_hybrid | recall_wikipedia | recall_graph | conflict_rate | retry_rate | EM_when_retried | avg_confidence | avg_llm_calls | avg_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval_closed_book | 500 | 24.2 | 28.5 | 24.6 | 0.0 |  |  |  |  |  |  |  | 1.0 | 0.7 |
| eval_single_rag | 500 | 37.2 | 42.5 | 40.4 | 70.8 | 70.8 |  |  |  |  |  |  | 1.0 | 8.0 |
| eval_full | 500 | 48.4 | 56.4 | 56.4 | 94.3 | 89.1 | 74.5 | 90.8 | 3.8 | 3.8 | 36.8 | 0.8 | 4.1 | 37.0 |

## By question type

| system | type | n | EM | F1 | Cover |
|---|---|---|---|---|---|
| eval_closed_book | bridge_comparison | 96 | 44.8 | 45.7 | 44.8 |
| eval_closed_book | comparison | 120 | 54.2 | 54.8 | 54.2 |
| eval_closed_book | compositional | 211 | 5.2 | 7.4 | 6.2 |
| eval_closed_book | inference | 73 | 2.7 | 23.9 | 2.7 |
| eval_single_rag | bridge_comparison | 96 | 54.2 | 55.7 | 58.3 |
| eval_single_rag | comparison | 120 | 77.5 | 79.7 | 78.3 |
| eval_single_rag | compositional | 211 | 14.7 | 19.2 | 19.0 |
| eval_single_rag | inference | 73 | 13.7 | 31.3 | 16.4 |
| eval_full | bridge_comparison | 96 | 43.8 | 43.8 | 43.8 |
| eval_full | comparison | 120 | 90.0 | 90.4 | 90.0 |
| eval_full | compositional | 211 | 37.9 | 49.1 | 55.5 |
| eval_full | inference | 73 | 16.4 | 38.1 | 20.5 |

### Confidence calibration: eval_full

| confidence | n | EM | F1 |
|---|---|---|---|
| [0.0, 0.4] | 95 | 10.5 | 17.2 |
| [0.4, 0.6] | 19 | 15.8 | 24.2 |
| [0.6, 0.8] | 42 | 28.6 | 44.1 |
| [0.8, 1.0] | 344 | 63.1 | 70.5 |