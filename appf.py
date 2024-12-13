import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.special import inv_boxcox
from datetime import datetime, timedelta
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
# Set up the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', st.secrets.get("GOOGLE_API_KEY"))
genai.configure(api_key=GOOGLE_API_KEY)
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to generate response from the model
def generate_response(prompt, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        # Include context from uploaded data in the prompt
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text  # Use 'text' attribute
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."


# Load SARIMA model
sarima_model = joblib.load('sarima_model.pkl')
# Load the dataset with a specified encoding
data = pd.read_csv('dataframe.csv', encoding='latin1')

# Load SARIMA model
sarima_model = joblib.load('sarima_model.pkl')

# Function to plot Box-Cox transformed graph
def plot_boxcox_graph(train_df, test_df, forecast_values_boxcox, confidence_intervals_boxcox):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_df['date'], train_df['actual_pickup_boxcox'], label='Actual Pickups (Train) - Box-Cox Transformed')
    ax.plot(test_df['date'], test_df['actual_pickup_boxcox'], label='Actual Pickups (Test) - Box-Cox Transformed')
    ax.plot(test_df['date'], forecast_values_boxcox, label='Forecasted Pickups (Box-Cox Transformed)')
    ax.fill_between(
        test_df['date'],
        confidence_intervals_boxcox.iloc[:, 0],
        confidence_intervals_boxcox.iloc[:, 1],
        color='gray', alpha=0.3, label='Confidence Interval'
    )
    ax.set_title('SARIMA Forecast (Box-Cox Transformed)')
    ax.legend()
    plt.xticks(rotation=45)
    return fig

# Function to plot graph with reversed Box-Cox transformation
def plot_original_graph(df, test_df, forecast_values_original):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['actual_pickup'], label='Actual Pickups')
    ax.plot(test_df['date'], forecast_values_original, label='Forecasted Pickups (Original Scale)')
    ax.set_title('SARIMA Forecast (Original Scale)')
    ax.legend()
    plt.xticks(rotation=45)
    return fig
# Function to generate exogenous variables
def generate_exog(start_date, days):
    """
    Generate exogenous values for the specified number of days.
    Replace this with your logic to fetch or estimate exog variables.
    """
    future_exog = {
        "scheduled_pickup": [100 + i * 2 for i in range(days)],
        "scheduled_pickup_lag_7": [90 + i for i in range(days)],
        "scheduled_pickup_lag_14": [80 + i for i in range(days)],
    }
    return pd.DataFrame(future_exog, index=pd.date_range(start=start_date, periods=days, freq="D"))

# Function to predict using SARIMA and plot
def predict_for_days(start_date, days):
    """
    Predict the total food hampers needed for a specified number of days and display results.
    """
    try:
        # Generate exogenous variables
        future_exog = generate_exog(start_date, days)

        # Forecast using SARIMA model
        predictions = sarima_model.forecast(steps=days, exog=future_exog)

        # Create a DataFrame for predictions
        forecast_dates = future_exog.index
        prediction_df = pd.DataFrame({"Date": forecast_dates, "Predicted Hampers": predictions})

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast_dates, predictions, label="Forecast", marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Food Hampers")
        ax.set_title("SARIMA Model Forecast")
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return prediction_df, fig
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

# Streamlit application
# Page 1: Dashboard
def dashboard():
        # Add an image
    st.image("downloads.png", use_column_width=True)

    st.subheader("💡 Project Overview:")
    inspiration = '''Project Overview We are collaborating on a machine learning project with a food hamper
    distribution company. The organization has shared their dataset with us and highlighted a number of challenges
    they face, such as resource allocation and meeting rising demand. After analyzing their needs, we identified that predicting
    the number of food hampers to be distributed in the near future could address several of these challenges. Our project will focus on
    developing a model to accurately forecast hamper distribution, enabling better planning and resource management for the organization.
    '''
    st.write(inspiration)
    st.subheader("Steps :")
    hello = ''' Here’s a concise breakdown of the steps we have done:
    1. Data Cleaning
    2. Data Visualizations
    3. ML Modelling
    4. Chat Bot
    '''
    st.write(hello)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Data Visualizations")
    st.markdown("""
    <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/b91808fe-0100-4e7f-94d4-957c4fea0c20/page/AtrGE" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
    """, unsafe_allow_html=True)

# Page 3: Machine Learning Modeling
# Streamlit application
def machine_learning_modeling():
    st.title("Food Hamper Forecasting")

    # Subsection: SARIMA Model for Food Hampers
    st.subheader("Food Hamper Forecasting (SARIMA Model)")

    # Input for start date
    start_date = st.date_input("Select the start date:", datetime.today())

    # Input for the number of days to forecast
    days = st.number_input("Enter the number of days to forecast:", min_value=1, step=1, value=1)

    if st.button("Predict Food Hampers"):
        # Call the prediction function with the selected start date and number of days
        predictions_df, fig = predict_for_days(start_date.strftime("%Y-%m-%d"), int(days))

        if predictions_df is not None:
            st.pyplot(fig)
            st.write("### Forecasted Food Hampers")
            st.write(predictions_df)
            total_hampers = predictions_df["Predicted Hampers"].sum()
            st.success(f"For {days} days starting from {start_date}, "
                       f"you will need approximately {int(total_hampers)} food hampers.")

# Page 4: Display SARIMA Forecast Graphs
def sarima_forecast_graphs():
    st.title("SARIMA Forecast Graphs")

    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Prepare train and test sets
    train_df = data.iloc[:int(len(data) * 0.8)]  # First 80% of the data
    test_df = data.iloc[int(len(data) * 0.8):]  # Last 20% of the data

    # Generate exogenous variables for test set
    future_exog = test_df[['scheduled_pickup', 'scheduled_pickup_lag_7', 'scheduled_pickup_lag_14']]

    # Forecast using SARIMA model
    forecast_values_boxcox = sarima_model.forecast(steps=len(test_df), exog=future_exog)
    confidence_intervals_boxcox = pd.DataFrame({
        0: forecast_values_boxcox - 0.1,  # Placeholder lower bound (adjust with actual)
        1: forecast_values_boxcox + 0.1   # Placeholder upper bound (adjust with actual)
    })

    # Reverse Box-Cox transformation (placeholder for actual lambda)
    forecast_values_original = np.exp(forecast_values_boxcox)  # Adjust transformation as needed

    # Plot the Box-Cox transformed graph
    st.subheader("Forecast (Box-Cox Transformed Data)")
    fig1 = plot_boxcox_graph(train_df, test_df, forecast_values_boxcox, confidence_intervals_boxcox)
    st.pyplot(fig1)

    # Plot the reversed transformation graph
    st.subheader("Forecast (Original Scale)")
    fig2 = plot_original_graph(data, test_df, forecast_values_original)
    st.pyplot(fig2)
# Page 5: Map 
def map():
    st.title("Map for Food Hamper Prediction.")
    st.markdown("""<iframe src="https://www.google.com/maps/d/u/0/embed?mid=1Uf7Agld8GzoH9-fzNNsUpmCN-0X8BEQ&ehbc=2E312F" width="640" height="480"></iframe>
    """, unsafe_allow_html=True)
#Page 6
# Streamlit app
def chatbot():
    st.title("Food Hamper Distribution Chatbot")
    st.write("Reading files from predefined paths...")

    # List of predefined file paths
    file_paths = ["mergedfoodandclientfinal.xlsx",
                  "NQ Report Template_DSN.pdf"
    ]

    # Prepare data context
    data_context = ""
    for file_path in file_paths:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data_context += f"\nData from {file_path}:\n{df.head(5).to_string()}\n"
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                data_context += f"\nData from {file_path}:\n{df.head(5).to_string()}\n"
            elif file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
                data_context += f"\nExtracted text from {file_path}:\n{text[:1000]}...\n"  # Limit to first 1000 characters
            st.success(f"Successfully processed {file_path}")
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about your project:", key="input")
    if st.button("Send"):
        if user_input and data_context:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input, data_context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        elif not data_context:
            st.error("No valid data context available. Please check the file paths.")

    for message in st.session_state.chat_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")
# Main App Logic
def main():
    st.sidebar.title("Food Hamper Prediction")
    app_page = st.sidebar.radio(
        "Select a Page", 
        ["Dashboard", "Data visualizations", "Sarima Model Predictions", "SARIMA Forecast Graphs", "Map for Food Hamper Prediction", "Chatbot"]
    )

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Data visualizations":
        exploratory_data_analysis()
    elif app_page == "Sarima Model Predictions":
        machine_learning_modeling()
    elif app_page == "SARIMA Forecast Graphs":
        sarima_forecast_graphs()
    elif app_page == "Map for Food Hamper Prediction":
        map()
    elif app_page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()
