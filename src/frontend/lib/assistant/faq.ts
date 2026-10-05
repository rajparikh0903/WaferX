export const FAQ: { group: string; items: string[] }[] = [
  { group: "Diagnosis", items: ["Why did this wafer fail?", "What is the main root cause?", "Which sensor affected the prediction most?", "Was this sensor normal or abnormal?", "What defect pattern was detected?", "Show similar historical wafers"] },
  { group: "Graph & Model Help", items: ["How do I understand the feature-importance graph?", "What does SHAP mean?", "Why is this sensor shown in red?", "What does failure risk mean?", "How is this prediction calculated?"] },
  { group: "Investigation", items: ["What should I investigate first?", "How is this wafer different from normal wafers?", "Did similar wafers fail before?", "Can changing this sensor reduce the predicted risk?"] },
];
