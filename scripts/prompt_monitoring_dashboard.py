#!/usr/bin/env python
"""
Prompt Monitoring Dashboard

This script creates a Streamlit dashboard for monitoring prompt performance metrics from MLflow.
It visualizes prompt versions, usage patterns, and performance metrics over time.

Usage:
    python prompt_monitoring_dashboard.py [--mlflow-uri URI]
"""

import os
import argparse
import pandas as pd
import numpy as np
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Prompt performance monitoring dashboard")
parser.add_argument(
    "--mlflow-uri",
    type=str,
    default=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"),
    help="MLflow tracking URI",
)
args = parser.parse_args()

# Set MLflow tracking URI
mlflow.set_tracking_uri(args.mlflow_uri)

# Initialize MLflow client
client = MlflowClient()


def load_prompt_runs():
    """Load all runs with prompt usage data from MLflow."""
    experiments = client.search_experiments()
    all_runs = []

    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="params.prompt_id != ''",
        )
        all_runs.extend(runs)

    # Convert to DataFrame
    runs_data = []
    for run in all_runs:
        # Extract basic info
        run_data = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": client.get_experiment(run.info.experiment_id).name,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
            "end_time": (
                datetime.fromtimestamp(run.info.end_time / 1000)
                if run.info.end_time
                else None
            ),
            "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
        }

        # Extract prompt info
        run_data["prompt_id"] = run.data.params.get("prompt_id", "unknown")
        run_data["prompt_version"] = run.data.params.get("prompt_version", "unknown")

        # Extract metrics
        run_data["tokens_used"] = run.data.metrics.get("tokens_used", np.nan)
        run_data["latency"] = run.data.metrics.get("latency", np.nan)
        run_data["accuracy"] = run.data.metrics.get("accuracy", np.nan)

        # Try to extract artifact data
        try:
            artifact_path = client.download_artifacts(
                run.info.run_id, "prompt_usage.json"
            )
            with open(artifact_path, "r") as f:
                prompt_usage = json.load(f)
                run_data["completion"] = prompt_usage.get("completion", "")
                run_data["inputs"] = str(prompt_usage.get("inputs", {}))
                run_data["metadata"] = str(prompt_usage.get("metadata", {}))
        except:
            pass

        runs_data.append(run_data)

    return pd.DataFrame(runs_data)


# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data():
    return load_prompt_runs()


# Streamlit app
st.title("Prompt Performance Monitoring")

# Load data
with st.spinner("Loading data from MLflow..."):
    df = get_data()

if df.empty:
    st.error("No prompt data found in MLflow. Make sure you have logged prompt usage.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# Filter by date
date_range = st.sidebar.date_input(
    "Date Range",
    value=(
        df["start_time"].min().date(),
        df["start_time"].max().date() + timedelta(days=1),
    ),
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[
        (df["start_time"].dt.date >= start_date)
        & (df["start_time"].dt.date <= end_date)
    ]
else:
    df_filtered = df

# Filter by prompt ID
prompt_ids = sorted(df_filtered["prompt_id"].unique())
selected_prompts = st.sidebar.multiselect(
    "Prompt IDs", prompt_ids, default=prompt_ids[:5] if len(prompt_ids) > 5 else prompt_ids
)

if selected_prompts:
    df_filtered = df_filtered[df_filtered["prompt_id"].isin(selected_prompts)]

# Overview metrics
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Prompts", len(df_filtered["prompt_id"].unique()))

with col2:
    st.metric("Total Versions", len(df_filtered[["prompt_id", "prompt_version"]].drop_duplicates()))

with col3:
    st.metric("Total Runs", len(df_filtered))

with col4:
    avg_tokens = df_filtered["tokens_used"].mean()
    st.metric("Avg. Tokens", f"{avg_tokens:.1f}" if not pd.isna(avg_tokens) else "N/A")

# Usage over time
st.header("Prompt Usage Over Time")
usage_df = (
    df_filtered.groupby([pd.Grouper(key="start_time", freq="D"), "prompt_id"])
    .size()
    .reset_index(name="count")
)

fig = px.line(
    usage_df,
    x="start_time",
    y="count",
    color="prompt_id",
    title="Prompt Usage by Day",
)
st.plotly_chart(fig, use_container_width=True)

# Performance by prompt version
st.header("Performance by Prompt Version")

# Create a view by prompt and version
performance_df = df_filtered.groupby(["prompt_id", "prompt_version"]).agg(
    runs=("run_id", "count"),
    avg_tokens=("tokens_used", "mean"),
    avg_latency=("latency", "mean"),
    avg_accuracy=("accuracy", "mean"),
).reset_index()

# Select a prompt for detailed view
selected_prompt = st.selectbox(
    "Select Prompt for Detailed View", 
    sorted(df_filtered["prompt_id"].unique())
)

prompt_df = performance_df[performance_df["prompt_id"] == selected_prompt]

if not prompt_df.empty:
    # Metrics by version
    fig = go.Figure()
    
    # Add a trace for tokens
    fig.add_trace(
        go.Bar(
            x=prompt_df["prompt_version"],
            y=prompt_df["avg_tokens"],
            name="Avg. Tokens",
            marker_color="royalblue",
        )
    )
    
    # Add a trace for latency on secondary axis
    fig.add_trace(
        go.Scatter(
            x=prompt_df["prompt_version"],
            y=prompt_df["avg_latency"],
            name="Avg. Latency (s)",
            marker_color="red",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    
    # Add accuracy if available
    if not prompt_df["avg_accuracy"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=prompt_df["prompt_version"],
                y=prompt_df["avg_accuracy"],
                name="Avg. Accuracy",
                marker_color="green",
                mode="lines+markers",
                yaxis="y3",
            )
        )
    
    # Update layout with multiple y-axes
    fig.update_layout(
        title=f"Performance Metrics by Version: {selected_prompt}",
        xaxis_title="Prompt Version",
        yaxis=dict(
            title="Avg. Tokens",
            titlefont=dict(color="royalblue"),
            tickfont=dict(color="royalblue"),
        ),
        yaxis2=dict(
            title="Avg. Latency (s)",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            anchor="x",
            overlaying="y",
            side="right",
        ),
        yaxis3=dict(
            title="Avg. Accuracy",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.9,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Prompt history
st.header("Prompt Version History")

# Get the prompt manager to load prompt content
try:
    from llm_ops_pipeline.utils.prompt_management import PromptManager
    
    prompt_manager = PromptManager()
    
    # Select a prompt to view its history
    selected_prompt_history = st.selectbox(
        "Select Prompt to View History", 
        sorted(df_filtered["prompt_id"].unique()),
        key="history_prompt"
    )
    
    # Get all versions of this prompt
    versions = sorted(
        df_filtered[df_filtered["prompt_id"] == selected_prompt_history]["prompt_version"].unique()
    )
    
    if versions:
        selected_version = st.selectbox("Select Version", versions)
        
        try:
            # Try to get the prompt content
            prompt_data = prompt_manager.get_prompt(selected_prompt_history)
            
            st.subheader(f"Prompt: {selected_prompt_history} (v{selected_version})")
            
            # Show metadata
            if "metadata" in prompt_data:
                with st.expander("Metadata"):
                    st.json(prompt_data["metadata"])
            
            # Show the prompt content
            st.code(prompt_data["content"], language="")
            
            # Show example usage
            with st.expander("Example Usage"):
                example_run = df_filtered[
                    (df_filtered["prompt_id"] == selected_prompt_history) &
                    (df_filtered["prompt_version"] == selected_version)
                ].iloc[0] if not df_filtered[
                    (df_filtered["prompt_id"] == selected_prompt_history) &
                    (df_filtered["prompt_version"] == selected_version)
                ].empty else None
                
                if example_run is not None and "inputs" in example_run and "completion" in example_run:
                    st.markdown("**Inputs:**")
                    st.code(example_run["inputs"])
                    st.markdown("**Completion:**")
                    st.code(example_run["completion"])
                else:
                    st.info("No example usage found for this prompt version.")
        
        except Exception as e:
            st.error(f"Failed to load prompt content: {e}")
            st.info("This could happen if the prompt exists in MLflow records but not in the local repository.")
    
except ImportError:
    st.warning("Prompt manager not available. Install the package to view prompt content.")

# Raw data view
with st.expander("View Raw Data"):
    st.dataframe(df_filtered)


if __name__ == "__main__":
    # This part is only executed when the script is run directly
    import sys
    
    if "--streamlit" in sys.argv:
        # Running with Streamlit
        import streamlit.cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
    else:
        # If running directly without --streamlit, print help
        print("This is a Streamlit dashboard. Run it with:")
        print(f"  streamlit run {sys.argv[0]} [--mlflow-uri URI]")
        print("Or:")
        print(f"  python {sys.argv[0]} --streamlit [--mlflow-uri URI]")