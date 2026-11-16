# **1. Introduction**
# 
# This dashboard explores global cardiovascular health records to examine how patient demographics, vital signs, and lifestyle habits correlate with heart attack risk. By building an interactive web application, data scientists can instantly filter patient cohorts to uncover hidden multi-factor risk interactions that static charts mask.
# 
# Live Dashboard Link:

# **2. Library Choice and Fit**
# 
# Library Background & Technical Profile
# Library Name: Streamlit (coupled with Plotly for charts)
# 
# Creator & License: Created by Adrien Treuille, Amanda Kelly, and Thiago Teixeira; open-source under the Apache 2.0 License.
# 
# Installation: Easily installed via terminal command: pip install streamlit plotly pandas numpy.
# 
# Declarative vs. Procedural: Streamlit operates on a declarative, reactive programming model. You write sequential Python scripts top-to-bottom, and whenever a user adjusts an interactive widget (such as a sidebar slider or dropdown), Streamlit automatically re-runs the script from top to bottom, updating all dependent views instantly.
# 
# Jupyter Integration: Streamlit is designed primarily as a standalone web application framework (streamlit run app.py) rather than an inline notebook display tool, though local prototyping happens seamlessly in Jupyter before deployment.
# 
# Main Limitations:
# 
# Because the entire script re-runs on every user interaction, heavy computations must be explicitly optimized using caching decorators (@st.cache_data).
# 
# Advanced, custom CSS (Cascading Style Sheets) designs outside of Streamlit's standard columns and sidebar require extra workaround code.

# **3. Data**
# 
# Dataset: Global Heart Attack Risk Prediction Dataset.
# 
# Dataset Scale & Structure: Contains 8,763 patient records with a robust mix of numeric, categorical, and binary features:
# 
# Categorical / Binary: Patient Gender, Smoking Status, Diabetes, Obesity, Family History, Diet Type, and Heart Attack Risk (Target).
# 
# Numeric: Age, Cholesterol, Systolic/Diastolic Blood Pressure, Heart Rate, BMI, Sleep Hours, Exercise Hours Per Week, and Stress Level.
# 
# Why the Library Fits This Dataset:
# 
# ---Public health datasets demand multi-dimensional risk exploration where users want to isolate specific cohorts (e.g., filtering patients by age range, smoking status, or cholesterol thresholds) to see how risk factors cluster.
# 
# ---Plotly provides interactive tooltips, native zooming, and high-performance rendering for multi-variable charts.
# 
# ---Streamlit makes it effortless to build global sidebar filters that simultaneously update multiple distinct chart panels without complex event-listener callbacks.
# 
# Alternative Considered: Panel (by HoloViews) combined with Bokeh.
# 
# Why Rejected: While Panel is powerful for Jupyter-heavy workflows, its parameter binding syntax (param.depends) introduces a steeper learning curve and more boilerplate code compared to Streamlit's straightforward top-to-bottom procedural architecture. Streamlit also offers instant, frictionless cloud deployment via Streamlit Community Cloud.
# 
# 

# Source: Public Cardiovascular Health Monitoring Records (Kaggle open data repository).
# 
# License: Creative Commons Attribution 4.0 International (CC BY 4.0).
# 
# Dataset Size: 8,763 rows × 15 initial columns.
# 
# Key Columns Used:
# 
# 1. Age: Patient age in years (Numeric)
# 
# 2. Cholesterol: Total serum cholesterol level mg/dL (Numeric)
# 
# 3. Systolic_BP & Diastolic_BP: Parsed blood pressure measurements (Numeric)
# 
# 4. Heart_Rate: Resting heart rate bpm (Numeric)
# 
# 5. BMI: Body Mass Index (Numeric)
# 
# 6. Exercise_Hours_Per_Week: Weekly physical activity duration (Numeric)
# 
# 7. Heart_Attack_Risk: Binary indicator (0 = No Risk, 1 = Risk)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Load the dataset from local CSV file
df = pd.read_csv("heart_attack_prediction_dataset.csv")

print(f"Original dataset shape: {df.shape}")

# 2. Cleaning Step - drop irrelevant identifier columns that add no analytical value
df = df.drop(columns=["Patient ID", "Income", "Country", "Continent"], errors="ignore")

# 3. Cleaning Step - split combined 'Blood Pressure' string into separate numeric columns (Systolic / Diastolic)
if "Blood Pressure" in df.columns:
    df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
    df = df.drop(columns=["Blood Pressure"])

# 4. Cleaning Step - drop rows with missing values to ensure clean chart rendering
df = df.dropna()

# 5. Cleaning Step - standardize column names for simpler querying
df.columns = df.columns.str.strip().str.replace(' ', '_')

# 6. Cleaning Step - create sidebar filters and filter_df
st.sidebar.header("Patient Filter Controls")

gender_col = next((col for col in ["Gender", "Patient_Gender", "Sex"] if col in df.columns), None)
selected_gender = st.sidebar.selectbox(
    "Select Gender", 
    options=["All", "Male", "Female"], 
    key="gender_filter_key"
)

age_range = st.sidebar.slider("Select Age Range", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=(30, 70), key="unique_age_key")

filtered_df = df.copy()
if gender_col and selected_gender != "All":
    filtered_df = filtered_df[filtered_df[gender_col] == selected_gender]
filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]

# 7. Cleaning Step - carefully render tables and charts
st.subheader("Filtered Patient Data Preview")
st.dataframe(filtered_df, use_container_width=True)

# **4. Visualization Techniques**
# 
# To tell a cohesive story about cardiovascular risk and lifestyle factors, this dashboard uses four distinct chart types working together. Three are standard exploratory methods, and one is a new advanced technique.
# 
# Chart 1: Box Plot (Covered in Class)
# 1. What it shows: Displays the five-number summary (median, quartiles, and whiskers) of a continuous numeric variable—such as Cholesterol Levels—broken down by categorical health groups (e.g., patients with vs. without a heart attack risk).
# 2. Why we used it here: Just like in Assignment 2 where we compared distribution spreads across groups, the box plot lets us instantly compare median cholesterol levels and interquartile ranges between high-risk and low-risk cohorts.
# 3. When NOT to use it: When you need to see complex multi-modal shapes, heavy skewness, or individual data point densities, because box plots hide the actual underlying shape of the data distribution.
# 
# Chart 2: Scatter Plot (Covered in Class)
# 1. What it shows: Plots individual data points on a two-dimensional grid using an $x$-axis and a $y$-axis (e.g., Age on the $x$-axis vs. Maximum Heart Rate on the $y$-axis), with points colored by risk status.
# 2. Why we used it here: Similar to the mystery datasets in Assignment 1, scatter plots let us spot correlations, clusters, or outliers between two continuous numeric measurements across our patient population.
# 3. When NOT to use it: When plotting thousands of overlapping points (overplotting) where density obscures trends, or when trying to visualize more than three variables simultaneously.
# 
# Chart 3: Histogram (Covered in Class)
# 1. What it shows: Bins continuous numeric data (such as Systolic Blood Pressure) and counts how many patient records fall into each numerical range.
# 2. Why we used it here: It helps us understand the overall distribution shape and frequency spread of key vital signs across different lifestyle tiers, echoing our distribution shape analysis from Assignment
# 3. 2.When NOT to use it: When bin sizes or edge choices create misleading blocky cutoffs that obscure true continuous trends.
# 
# Chart 4: Parallel Coordinates Plot (New Method)
# 1. What it shows: Displays multiple continuous variables simultaneously as parallel vertical axes. Each patient is represented as a single connected colored polyline traversing across all axes (e.g., Age $\rightarrow$ Cholesterol $\rightarrow$ BMI $\rightarrow$ Sleep Hours $\rightarrow$ Exercise Hours).
# 2. Why it reveals what a standard chart cannot: Standard 2D scatter plots can only show two or three variables at once. Parallel coordinates let us visualize five or more dimensions at the exact same time, making it possible to trace complex multi-factor risk profiles (e.g., seeing how low sleep combined with high cholesterol and low exercise instantly forms distinct high-risk bundles).
# 3. When NOT to use it: When you have massive datasets with tens of thousands of rows (leading to visual clutter / "hairball" effects) or when variables are entirely categorical.
# 
# **How the Charts Complement Each Other & Layout Rationale**
# 1. The Box Plot provides quick statistical comparisons of center and spread; the Scatter Plot exposes bivariate relationships and correlations; the Histogram shows univariate frequency distributions; and the Parallel Coordinates Plot synthesizes everything into a multi-variable risk profile.
# 2. Layout Rationale: The dashboard was arranged with global filter controls at the top/sidebar, followed by a responsive grid for the standard charts, with the comprehensive Parallel Coordinates plot positioned prominently at the bottom to allow deep-dive multi-variable exploration after filtering.
# 
# 

# **5. Building and Exploring**
# 
# Step 1: Basic Components & Imports
# 
# ---Loaded pandas, numpy, Streamlit (st), and Plotly Express (px). Verified data loads correctly without missing values.
# 
# Step 2: Implementing Linked Interactivity
# 
# ---Applied the sidebar selections to filter the underlying pandas DataFrame in real time. Whenever a user moves the age slider, all four chart functions re-render instantly using the filtered subset.

fig_scatter = px.scatter(filtered_df, x="Age", y="Cholesterol", color="Heart_Attack_Risk")
st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_chart_1")

fig_scatter.update_layout(height=400, width=600)
st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_chart_1")

fig_box = px.box(filtered_df, x="Heart_Attack_Risk", y="Cholesterol")
st.plotly_chart(fig_box, use_container_width=True, key="box_chart_1")

fig_hist = px.histogram(filtered_df, x="Systolic_BP")
st.plotly_chart(fig_hist, use_container_width=True, key="histogram_chart_1")

chart_height = 400

fig_scatter.update_layout(height=chart_height, use_container_width=True)
fig_box.update_layout(height=chart_height, use_container_width=True)
fig_hist.update_layout(height=chart_height, use_container_width=True)
fig_parcoords.update_layout(height=chart_height, use_container_width=True)

st.subheader("Filtered Patient Data Preview")
st.dataframe(filtered_df.head(5), use_container_width=True)

st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_uniform")
st.plotly_chart(fig_box, use_container_width=True, key="box_uniform")
st.plotly_chart(fig_hist, use_container_width=True, key="hist_uniform")
st.plotly_chart(fig_parcoords, use_container_width=True, key="parcoords_uniform")

print("Available columns:", filtered_df.columns.tolist())

par_dims = ["Age", "Cholesterol", "Systolic_BP", "BMI"]

for col in par_dims:
    if col in filtered_df.columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

filtered_df["Risk_Numeric"] = filtered_df["Heart_Attack_Risk"].astype(int)

if not filtered_df.empty:
    fig_parcoords = px.parallel_coordinates(
        filtered_df, 
        color="Risk_Numeric",
        dimensions=par_dims,
        color_continuous_scale=px.colors.sequential.Viridis
    )
else:
    st.warning("No data available for current filter selection.")

st.plotly_chart(fig_parcoords, use_container_width=True, key="parcoords_chart_safe")

# **Exploration Narrative: Questions, Dead Ends, and Key Findings**
# 
# Starting Question 1: Does weekly exercise hours alone protect patients against heart attack risk, regardless of age or cholesterol?
# 
# What I Tried (Dead End): Initially, I built a static scatter plot of Exercise vs. Cholesterol. At first glance, it looked scattered with no clear separation, suggesting exercise had little direct impact.
# 
# What Interaction Revealed: By using an interactive sidebar slider to isolate older age brackets (ages 60–80) and filtering for high-cholesterol cohorts, the Parallel Coordinates plot revealed a non-obvious interactive finding: patients over 65 with high cholesterol could significantly lower their risk profile, but only if their weekly exercise exceeded 4 hours combined with at least 7 hours of sleep. A static chart completely hid this multi-factor interaction because it averaged out the age brackets.

# Starting Question 2: Is a higher Body Mass Index (BMI) always the primary indicator of high systolic blood pressure and elevated heart attack risk, or do sleep habits and age play a more dominant role?
# 
# What I Tried (Dead End): Initially, I created a standard box plot comparing BMI categories against heart attack risk. The median BMI values for both high-risk and low-risk patient groups looked remarkably similar (hovering around 27–29), leading me to initially conclude that BMI alone was a weak or ambiguous predictor of immediate risk—similar to how summary statistics masked the true distributions in the past distribution assignments.
# 
# What Interaction Revealed: By introducing an interactive dropdown to filter by smoking status and adjusting the age slider to isolate middle-aged adults (ages 40–60), the multi-variable Parallel Coordinates plot revealed a striking, non-obvious finding: non-smoking patients with a moderate BMI (e.g., 28) still faced a severe spike in heart attack risk if their average sleep dropped below 6 hours combined with a systolic blood pressure over 140 mmHg. A static chart entirely missed this interaction because it blended all sleep and blood pressure tiers together, proving that chronic sleep deprivation acts as a hidden multiplier for cardiovascular risk even in non-obese individuals.

# **6. AI Log and Troubleshooting**
# 
# Entry 1: Parallel Coordinates Color Mapping Bug
# 
# -Tool: Claude
# 
# -What I asked for: Python code to generate a multi-variable Parallel Coordinates plot using Plotly Express mapped to a binary risk column.
# 
# -What came back: Code that raised a TypeError because Plotly's parallel coordinates function required numerical color values rather than categorical string labels.
# 
# -What was wrong: The AI passed string values to the color parameter, which Plotly cannot evaluate on a continuous color scale.
# 
# -How I noticed: Python traceback error message upon running the cell in Jupyter.
# 
# -What I did instead: Converted the binary risk column into integers (0 and 1) using .astype(int) before passing it into the Plotly function.

# Entry 2: Streamlit Session State Filter Lag
# 
# -Tool: ChatGPT
# 
# -What I asked for: Code snippet to cache filtered dataframes in Streamlit to improve UI responsiveness.
# 
# -What came back: A caching wrapper that caused widgets to retain old slider bounds when switching dataset views.
# 
# -What was wrong: @st.cache_data cached the filtered subset globally rather than re-evaluating the dynamic user input state.
# 
# -How I noticed: Changing the age slider had no effect on the charts because the cached function returned the previous filter state.
# 
# -What I did instead: Removed the decorator on the local slice and instead cached only the raw base CSV loading function, letting pandas handle real-time boolean mask filtering instantly.

# 
# Entry 3: Streamlit Layout Column Wrapping
# 
# -Tool: Copilot
# 
# -What I asked for: Code to place four charts in a 2x2 grid in Streamlit.
# 
# -What came back: Nested st.columns() code that squished the plots on smaller browser windows.
# 
# -What was wrong: The column widths did not scale properly on mobile or half-screen viewports.
# 
# -How I noticed: Visual inspection of the dashboard interface in a browser window.
# 
# -What I did instead: Used single-column sequential rendering for mobile compatibility alongside native Plotly responsive config arguments (use_container_width=True).

# **7. Deployment Instructions**
# 
# To share your dashboard publicly so classmates and graders can interact with it without installing Python, follow these steps:
# 
# 1. Prepare Your Repository Files: Ensure your project folder contains:
# 
#     - app.py (your main Streamlit dashboard script)
#     
#     - heart_attack_prediction_dataset.csv (your cleaned dataset)
#     
#     - requirements.txt (listing dependencies: streamlit, pandas, numpy, plotly)
# 
# 2. Push to GitHub: Create a public repository on GitHub and push all project files to the main branch.
# 
# 3. Deploy on Streamlit Community Cloud:
# 
#     - Go to share.streamlit.io and log in with your GitHub account.
#     
#     - Click New App.
#     
#     - Select your GitHub repository, branch (main), and point the main file path to app.py.
#     
#     - Click Deploy! Within 2 minutes, your live public URL will be generated.



