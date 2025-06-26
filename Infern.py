# Infern 
# Author: Jay Pandya

import gradio as gr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

stored = {"df": None, "figs": {}, "last_results": None}

def load_data(file):
    try:
        df = pd.read_csv(file.name, encoding='utf-8', on_bad_lines='skip')
    except:
        df = pd.read_csv(file.name, encoding='ISO-8859-1', on_bad_lines='skip')
    stored["df"] = df
    stored["figs"] = {}
    stored["last_results"] = None
    return df.head()

def run_automl_multi_target(df, targets, fast_mode=True):
    results = {}
    for target in targets:
        if target not in df.columns:
            results[target] = "❌ Target column not found."
            continue

        try:
            X = df.drop(columns=[target])
            y = df[target]
            X = pd.get_dummies(X)
            X = X.select_dtypes(include=[np.number])
            X.fillna(0, inplace=True)
            X = X.loc[:, (X.std() > 0).values]
            if X.shape[1] > 100:
                X = X.iloc[:, :100]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            if X_train.shape[1] == 0 or len(y_train) == 0:
                results[target] = "⚠️ No valid features or target values to train on."
                continue

            if y.nunique() <= 10:
                model = LazyClassifier(verbose=0, ignore_warnings=True)
            else:
                model = LazyRegressor(verbose=0, ignore_warnings=True)

            models, _ = model.fit(X_train, X_test, y_train, y_test)
            if models.empty:
                results[target] = "⚠️ No models could be trained — check your data."
                continue

            top_models = models.sort_values(by=models.columns[1], ascending=False).head(5)
            results[target] = top_models.reset_index()
        except Exception as e:
            results[target] = f"⚠️ Error: {str(e)}"
    return results

def run_automl(target, mode):
    df = stored["df"]
    if not target or (target not in df.columns and "," not in target):
        return pd.DataFrame({"Error": [f"{target} not found."]})

    if "," in target:
        targets = [t.strip() for t in target.split(",") if t.strip() in df.columns]
        if not targets:
            return pd.DataFrame({"Error": ["No valid target columns found."]})
        result = run_automl_multi_target(df, targets, fast_mode=(mode == "Fast Mode"))
        output = {}
        for k, v in result.items():
            if isinstance(v, pd.DataFrame):
                output[k] = v.to_dict(orient="records")
            else:
                output[k] = v
        stored["last_results"] = (target, output)
        return output

    df_clean = df.dropna(subset=[target]).copy()
    y = df_clean[target]
    X = df_clean.drop(columns=[target])
    X = pd.get_dummies(X)
    X = X.select_dtypes(include=np.number).fillna(0)
    X = X.loc[:, (X.std() > 0).values]
    if X.shape[1] > 100:
        X = X.iloc[:, :100]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if X_train.shape[1] == 0 or len(y_train) == 0:
        return pd.DataFrame({"Error": ["⚠️ No valid features or target values to train on."]})

    if mode == "LazyPredict":
        model = LazyClassifier(verbose=0, ignore_warnings=True) if y.nunique() <= 10 else LazyRegressor(verbose=0, ignore_warnings=True)
        results, _ = model.fit(X_train, X_test, y_train, y_test)
        if results.empty:
            return pd.DataFrame({"Error": ["⚠️ No models could be trained — check your data."]})
        stored["last_results"] = (target, results.head(5))
        return results.reset_index().head(5)
    else:
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results = pd.DataFrame({"Model": ["LogisticRegression"], "Accuracy": [round(acc, 3)]})
        stored["last_results"] = (target, results)
        return results

def generate_automl_recommendations():
    df = stored["df"]
    result = stored.get("last_results")
    if not result:
        return "⚠️ Run AutoML first to generate recommendations."

    target, output = result
    if isinstance(output, dict):
        first_key = next(iter(output))
        df_top = pd.DataFrame(output[first_key])
    elif isinstance(output, pd.DataFrame):
        df_top = output
    else:
        return "⚠️ No valid results for recommendation."

    summary = []

    model_name = df_top.iloc[0][0] if isinstance(df_top.iloc[0][0], str) else "Top Model"
    summary.append(f"✅ Best Model Found: Use **{model_name}** for best performance.\n")

    tech_keywords = ["id", "code", "serial", "index", "number"]
    def is_technical(col): return any(k in col.lower() for k in tech_keywords)

    text_cols = [col for col in df.select_dtypes(include='object').columns if not is_technical(col)]
    for col in text_cols[:3]:
        if df[col].nunique() > 1:
            top_val = df[col].value_counts().idxmax()
            count = df[col].value_counts().max()
            insight = f"📌 In **{col}**, the most frequent value is **{top_val}** with **{count} occurrences**. Explore its business significance."
            summary.append(insight)

    num_cols = [col for col in df.select_dtypes(include='number').columns if not is_technical(col)]
    for col in num_cols[:1]:
        avg_val = df[col].mean()
        insight = f"📊 The average value in **{col}** is **{round(avg_val, 2)}**, useful for setting benchmarks or KPIs."
        summary.append(insight)

    return "\n\n".join(summary)

def generate_chart(chart_type, column):
    df = stored["df"]
    if column not in df.columns:
        return go.Figure()
    vc = df[column].value_counts().reset_index()
    vc.columns = [column, 'count']
    if chart_type == "Bar":
        fig = px.bar(vc, x=column, y='count')
    elif chart_type == "Column":
        fig = px.histogram(df, x=column)
    elif chart_type == "Pie":
        fig = px.pie(df, names=column)
    elif chart_type == "Donut":
        fig = px.pie(df, names=column, hole=0.5)
    else:
        fig = go.Figure()
    stored["figs"]["chart"] = fig
    return fig

def generate_choropleth(location_col, value_col):
    df = stored["df"]
    fig = px.choropleth(df, locations=location_col, locationmode="country names", color=value_col)
    stored["figs"]["choropleth"] = fig
    return fig

def generate_forecast(date_col, value_col):
    df = stored["df"]
    if date_col not in df.columns or value_col not in df.columns:
        return go.Figure()
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    df_grouped = df.groupby(date_col)[value_col].mean().reset_index()
    fig = px.line(df_grouped, x=date_col, y=value_col, title=f"Trend of {value_col} Over Time")
    stored["figs"]["forecast"] = fig
    return fig

def update_dropdowns():
    if stored["df"] is not None:
        cols = stored["df"].columns.tolist()
        return [gr.update(choices=cols) for _ in range(6)]
    else:
        return [gr.update(choices=[]) for _ in range(6)]

with gr.Blocks(css="body { font-family: 'Segoe UI'; }") as app:
    gr.Markdown("""
    <div style='text-align: center; font-size: 28px; font-weight: bold; margin-bottom: -10px;'>
    Infern — AutoML Intelligence Platform
    </div>
    <div style='text-align: center; font-size: 16px; font-style: italic; margin-bottom: 20px;'>
    From data to insight, in a single click.
    </div>
    """)

    with gr.Tab("📁 Upload & Preview"):
        file_input = gr.File()
        preview = gr.Dataframe()
        file_input.change(fn=load_data, inputs=file_input, outputs=preview)

    with gr.Tab("⚙️ AutoML"):
        target_dropdown = gr.Textbox(label="Target Column(s) (comma-separated for multiple)")
        mode_dropdown = gr.Radio(["LazyPredict", "Fast Mode"], value="LazyPredict", label="Mode")
        run_btn = gr.Button("Run AutoML")
        results = gr.JSON()
        run_btn.click(fn=run_automl, inputs=[target_dropdown, mode_dropdown], outputs=results)

    with gr.Tab("📊 Insights"):
        chart_type = gr.Dropdown(["Bar", "Column", "Pie", "Donut"], label="Chart Type")
        column_dropdown = gr.Dropdown(label="Select Column")
        chart_btn = gr.Button("Generate Chart")
        chart_output = gr.Plot()
        chart_btn.click(fn=generate_chart, inputs=[chart_type, column_dropdown], outputs=chart_output)

    with gr.Tab("🌍 Choropleth & Forecast"):
        location_col = gr.Dropdown(label="Country Column")
        value_col = gr.Dropdown(label="Value Column")
        choropleth_btn = gr.Button("Generate Choropleth")
        choropleth_output = gr.Plot()
        choropleth_btn.click(fn=generate_choropleth, inputs=[location_col, value_col], outputs=choropleth_output)

        date_col = gr.Dropdown(label="Date Column")
        trend_val = gr.Dropdown(label="Value Column")
        forecast_btn = gr.Button("Generate Forecast")
        forecast_output = gr.Plot()
        forecast_btn.click(fn=generate_forecast, inputs=[date_col, trend_val], outputs=forecast_output)

    with gr.Tab("🧠 AutoML Recommendations"):
        rec_output = gr.Markdown()
        generate_btn = gr.Button("Generate Business Insights")
        generate_btn.click(fn=generate_automl_recommendations, outputs=rec_output)

    gr.Markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: #666666;
        font-size: 13px;
    }
    </style>
    <div class='footer'>
        © 2025 Jay Pandya | All rights reserved.
    </div>
    """)

    preview.change(fn=update_dropdowns, inputs=None,
        outputs=[column_dropdown, column_dropdown, location_col, value_col, date_col, trend_val])

app.launch()
