# Problem Statement

## Background

The problem belongs to the **semiconductor manufacturing and yield optimization** domain, particularly advanced nodes such as **3nm and 5nm**. Modern semiconductor fabs generate large volumes of data from equipment sensors, process parameters, and wafer defect images. Although this data contains valuable signals about yield loss and process failures, the information is spread across many sources, making it difficult for engineers to identify the true reasons behind defects and yield degradation.

## The Problem

At 3nm/5nm nodes, even a **1% drop in yield can cost tens of millions of dollars per month**. The underlying causes can be hidden across thousands of equipment sensors, process parameters, and wafer defect images. Engineers often need to manually correlate these different data sources to determine why yield dropped, which can take weeks and delay corrective action.

Furthermore, defect patterns can repeat across different shifts because there is no systematic learning loop connecting **failure detection, root-cause identification, process correction, and historical outcomes**.

## Who is Affected

The primary users are **semiconductor process and yield engineers** responsible for monitoring manufacturing quality, investigating yield loss, identifying process abnormalities, and determining corrective actions.

They must analyze large numbers of sensor readings, process parameters, historical failures, and wafer defect patterns to investigate a single manufacturing issue.

## Why It Matters

Yield loss directly affects semiconductor manufacturing revenue and production efficiency.

- A **1% yield reduction at advanced nodes can represent tens of millions of dollars in monthly losses**.
- Manual investigation can take **weeks**, increasing the financial impact of every unresolved issue.
- Repeated defect patterns may continue across shifts when previous failure knowledge is not systematically reused.
- Engineers need not only failure detection, but also **evidence-based root-cause analysis and evaluation of possible corrective actions**.

Reducing the time from detection to diagnosis and corrective decision can therefore have a significant operational and financial impact.

## Why Existing Solutions Fall Short

Existing manufacturing systems can provide large amounts of sensor, process, and defect data, and individual machine-learning models can predict failures or detect anomalies. However, these capabilities are often fragmented.

Engineers still need to manually correlate:

- Process and equipment sensor behaviour
- Anomaly detection results
- Root-cause evidence
- Historical failure patterns
- Wafer defect images
- Potential process changes

This makes the investigation process slow and limits systematic learning from previous failures.

**YieldTwin addresses this gap by combining prediction, anomaly detection, explainable root-cause analysis, wafer-image intelligence, historical evidence, and what-if analysis into a single engineering decision-support workflow.**