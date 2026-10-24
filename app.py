
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="📚"
)

st.title("📚 Student Marks Predictor")
st.write("Predict marks using Machine Learning!")

# Sample dataset
X = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 5]
).reshape(-1, 1)

y = np.array(
    [30, 38, 45, 50, 58, 65, 72, 78, 85, 92, 40, 60]
)

# Train the model
model = LinearRegression()
model.fit(X, y)

st.subheader("Make a prediction")

# Interactive slider
hours = st.slider(
    "How many hours did you study?",
    min_value=1,
    max_value=10,
    value=6
)

# Predict button
if st.button("Predict Marks 🚀"):
    prediction = model.predict([[hours]])[0]
    prediction = np.clip(prediction, 0, 100)

    st.success(f"Predicted marks: {prediction:.1f}/100")
    st.caption(
        "This is an estimate from sample data, not a guaranteed result."
    )

# Show the dataset
with st.expander("View sample training data"):
    st.dataframe({
        "Study Hours": X.flatten(),
        "Marks": y
    })

# Plot the regression line
st.subheader("📈 Linear Regression Graph")

fig, ax = plt.subplots()

ax.scatter(X.flatten(), y, label="Sample data")

line_X = np.linspace(1, 10, 100).reshape(-1, 1)
line_y = model.predict(line_X)

ax.plot(line_X.flatten(), line_y, label="Regression line")

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
ax.legend()
ax.grid(True)

st.pyplot(fig)
