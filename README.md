# ML Assignment 2 — Wall-Following Robot Navigation

## a. Problem statement

Build and compare five classical classification models that predict the next navigation action of a SCITOS G5 mobile robot from 24 ultrasound sensor readings, then expose the saved models through a Streamlit app that evaluates uploaded test CSV data **without retraining**.

## b. Dataset description

| Item | Detail |
|------|--------|
| Name | Wall-Following Robot Navigation Data (UCI) |
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data) / [Kaggle](https://www.kaggle.com/datasets/uciml/wall-following-robot) |
| Robot | SCITOS G5 following a wall clockwise for 4 rounds |
| Sensors | 24 ultrasound units around the robot waist (~15° apart), sampled at 9 Hz |
| Variant used | `sensor_readings_24` (raw US1–US24 + class) |
| Instances | 5,456 |
| Features | 24 numeric distances |
| Classes | Move-Forward (2205), Sharp-Right-Turn (2097), Slight-Right-Turn (826), Slight-Left-Turn (328) |

The collection was designed to show that wall-following is a **non-linearly separable** classification task: linear models struggle, while non-linear models perform better. Class imbalance is material (Slight-Left-Turn is rare).

Clean CSV: [`data/sensor_readings_24.csv`](data/sensor_readings_24.csv). Held-out test split: [`data/test_data.csv`](data/test_data.csv).

## c. GitHub repository link and Streamlit App URL

[Github repo here](https://github.com/SreenathSaikumar/ML-assignment-2)
[Streamlit app deployed here](https://ml-assignment-2-sreenath.streamlit.app)

## d. Models used

Stratified 80/20 split (`random_state=42`). Metrics on the held-out test set (weighted averages for multiclass Precision/Recall/F1; AUC = OvR):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6712 | 0.8777 | 0.7250 | 0.6712 | 0.6782 | 0.5482 |
| Decision Tree | 0.9890 | 0.9921 | 0.9891 | 0.9890 | 0.9890 | 0.9834 |
| KNN | 0.8599 | 0.9518 | 0.8597 | 0.8599 | 0.8596 | 0.7876 |
| Naive Bayes | 0.4973 | 0.7832 | 0.6196 | 0.4973 | 0.5019 | 0.3630 |
| Support Vector Machine | 0.8709 | 0.9599 | 0.8836 | 0.8709 | 0.8720 | 0.8139 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reasonable AUC but weaker accuracy/F1 — consistent with the known non-linear nature of the task. |
| Decision Tree | Highest scores across metrics; non-linear axis-aligned rules fit the sensor geometry well (note: random splits on sequential robot data can look optimistic). |
| KNN | Strong after scaling; competitive F1 without assuming a parametric form. |
| Naive Bayes | Weakest overall; Gaussian independence assumption fits poorly with correlated neighbouring sensors. |
| Support Vector Machine | Strong non-linear margin (RBF + calibration); second-best MCC/F1 trade-off after the tree. |
| **Overall winner** | **Decision Tree** on this held-out split (best Accuracy / F1 / MCC / AUC). |

## Project layout

```text
data/          # sensor_readings_24.csv, test_data.csv
model/         # saved .joblib pipelines + metrics
src/           # dataset/train/metrics helpers + Streamlit app.py
notebooks/     # EDA & training notebook
```

## Quick start

```bash
uv sync
uv run python -m src.train
uv run streamlit run src/app.py
```

Regenerate `requirements.txt`:

```bash
uv export --locked --no-default-groups --no-emit-project --format requirements-txt -o requirements.txt
```
