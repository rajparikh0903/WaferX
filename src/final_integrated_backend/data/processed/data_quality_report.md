# SECOM data-quality report

- Rows x columns: **1567 x 590**
- Duplicate rows: 0; non-numeric cells: 0; infinite cells: 0
- Missing cells: 4.54%; columns with any missing: 538; columns > 50% missing: 28; rows with any missing: 1567
- Constant features: 116; near-zero-variance: 6
- Target: PASS 1463 / FAIL 104 (fail rate 6.64%, imbalance 14.1:1)
- Timestamps: 2008-07-19T11:55:00 -> 2008-10-17T06:07:00; invalid 0; monotonic True

## Fail rate by time chunk

| chunk | from | to | n | fail rate |
|---|---|---|---|---|
| 1 | 2008-07-19T11:55:00 | 2008-08-19T18:30:00 | 314 | 0.140 |
| 2 | 2008-08-19T18:30:00 | 2008-09-01T08:18:00 | 314 | 0.067 |
| 3 | 2008-09-01T09:06:00 | 2008-09-20T06:08:00 | 313 | 0.035 |
| 4 | 2008-09-20T07:21:00 | 2008-10-02T20:54:00 | 313 | 0.035 |
| 5 | 2008-10-02T21:32:00 | 2008-10-17T06:07:00 | 313 | 0.054 |

## Split decision

temporal: samples are time-ordered, production data arrives in time order, and fail rate varies across time chunks (0.035-0.140), so a random split would leak future process behaviour into training and give optimistic metrics.