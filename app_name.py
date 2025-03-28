import streamlit as st
import random

# Page Configuration
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")

# Define lung cancer information
lung_cancer_data = {
    "Adenocarcinoma": {
        "description": "Adenocarcinoma is the most common type of lung cancer in non-smokers. It originates in the mucus-producing glands and primarily affects the outer lung regions. It is often detected at an advanced stage and can spread to lymph nodes or distant organs. Symptoms include a persistent cough, shortness of breath, and chest pain.",
        "medicines": ["Gefitinib", "Erlotinib", "Osimertinib", "Bevacizumab", "Afatinib", "Dacomitinib"],
        "treatments": ["Chemotherapy", "Targeted Therapy", "Immunotherapy", "Surgery"],
        "exercises": ["Deep breathing exercises", "Walking 30 mins daily", "Pulmonary rehabilitation", "Light resistance training", "Yoga"],
        "hospitals": ["Johns Hopkins Cancer Center", "MD Anderson Cancer Center", "Mayo Clinic", "Tata Memorial Hospital"],
        "treatment_duration": "6 months to 2 years (depends on stage & response to treatment)",
        "diet": [
            "✅ High-protein foods: Chicken, fish, tofu, eggs",
            "✅ Leafy green vegetables: Spinach, kale, broccoli",
            "✅ Healthy fats: Olive oil, nuts, avocados",
            "✅ Hydration: 8-10 glasses of water daily",
            "❌ Avoid processed foods, red meat, and excessive sugar"
        ]
    },
    "Squamous Cell Carcinoma": {
        "description": "Squamous Cell Carcinoma is linked to smoking and occurs in the bronchial tubes. It can cause airway obstruction, leading to persistent coughing, chest pain, and difficulty breathing. This type of cancer is treatable if detected early.",
        "medicines": ["Cisplatin", "Carboplatin", "Nivolumab", "Pembrolizumab", "Paclitaxel", "Gemcitabine"],
        "treatments": ["Radiation Therapy", "Surgery", "Immunotherapy", "Targeted Therapy"],
        "exercises": ["Lung capacity exercises", "Yoga for lung health", "Cardio workouts", "Stretching"],
        "hospitals": ["Memorial Sloan Kettering Cancer Center", "Cleveland Clinic", "Stanford Cancer Institute"],
        "treatment_duration": "4 months to 18 months (depends on severity)",
        "diet": [
            "✅ High-antioxidant foods: Berries, nuts, dark chocolate",
            "✅ Fiber-rich foods: Whole grains, oats, lentils",
            "✅ Lean proteins: Turkey, fish, eggs",
            "❌ Avoid smoking, alcohol, and junk food"
        ]
    },
    "Small Cell Lung Cancer": {
        "description": "Small Cell Lung Cancer is an aggressive cancer that grows rapidly and spreads to other organs. It is mostly diagnosed in heavy smokers. Common symptoms include unexplained weight loss, chronic cough, hoarseness, and fatigue.",
        "medicines": ["Etoposide", "Topotecan", "Irinotecan", "Atezolizumab", "Durvalumab", "Cyclophosphamide"],
        "treatments": ["Chemotherapy", "Radiation Therapy", "Immunotherapy", "Surgery (in rare cases)"],
        "exercises": ["Breathing techniques", "Aerobic exercises", "Physical therapy", "Tai Chi"],
        "hospitals": ["Duke Cancer Institute", "UCLA Health", "Apollo Hospitals"],
        "treatment_duration": "3 months to 1.5 years (depends on response)",
        "diet": [
            "✅ Omega-3 rich foods: Salmon, walnuts, flaxseeds",
            "✅ Calcium-rich foods: Dairy products, almonds, tofu",
            "✅ Hydration: Herbal teas, fresh juices, water",
            "❌ Avoid spicy and fried foods"
        ]
    },
    "Large Cell Lung Cancer": {
        "description": "Large Cell Lung Cancer is an aggressive type of lung cancer that can develop in any part of the lung. It has a high growth rate, making early detection critical. Symptoms include coughing up blood, wheezing, and extreme fatigue.",
        "medicines": ["Docetaxel", "Paclitaxel", "Pemetrexed", "Vinorelbine", "Bevacizumab"],
        "treatments": ["Surgery", "Chemotherapy", "Targeted Therapy", "Radiation Therapy"],
        "exercises": ["Breathing exercises", "Light resistance training", "Stretching", "Swimming"],
        "hospitals": ["Massachusetts General Hospital", "Mayo Clinic", "AIIMS, India"],
        "treatment_duration": "5 months to 2 years",
        "diet": [
            "✅ Lean proteins: Fish, chicken, lentils",
            "✅ Healthy fats: Olive oil, nuts, seeds",
            "✅ Antioxidant-rich foods: Berries, green tea, turmeric",
            "❌ Reduce processed foods and sugary drinks"
        ]
    }
}

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Lung Cancer Info", "Why Detect Lung Cancer?", "Find Hospitals"])

# Home Page - Lung Cancer Detection
if page == "Home":
    st.title("🫁 Lung Cancer Detection System")
    st.subheader("Upload a Chest X-ray or CT Scan to Detect Lung Cancer")

    uploaded_file = st.file_uploader("Upload a lung scan image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Randomly assign a cancer type (replace this with actual ML model prediction later)
        cancer_type = random.choice(list(lung_cancer_data.keys()))

        # Show detected details
        data = lung_cancer_data[cancer_type]
        st.markdown(f"## **Detected Cancer Type: {cancer_type}**")
        st.write(f"**📝 Description:** {data['description']}")
        st.write(f"**💊 Recommended Medicines:** {', '.join(data['medicines'])}")
        st.write(f"**🩺 Treatments:** {', '.join(data['treatments'])}")
        st.write(f"**🏋️ Exercise Suggestions:** {', '.join(data['exercises'])}")
        st.write(f"**🏥 Recommended Hospitals:** {', '.join(data['hospitals'])}")
        st.write(f"**⏳ Estimated Treatment Duration:** {data['treatment_duration']}")
        st.write(f"**🥗 Diet Plan:**")
        for item in data['diet']:
            st.write(item)

# Lung Cancer Information Page
elif page == "Lung Cancer Info":
    st.title("📖 Lung Cancer Information")
    st.write("### What is Lung Cancer?")
    st.write("""
        Lung cancer is one of the most common cancers worldwide and is caused by abnormal cell growth in the lungs.
        It can be categorized into **non-small cell lung cancer (NSCLC)** and **small cell lung cancer (SCLC)**.
        
        **Symptoms of Lung Cancer:**
        - Persistent cough
        - Shortness of breath
        - Fatigue
        - Chest pain
        - Unexplained weight loss
    """)

# Find Hospitals Page
elif page == "Find Hospitals":
    st.title("🏥 Find Hospitals Near You")
    st.write("🔎 Enter your location to find the best hospitals nearby.")
    location = st.text_input("Enter your city/country:")
    if location:
        st.write(f"Searching for top lung cancer hospitals in {location}...")

# Footer
st.markdown("---")
st.markdown("© 2025 Lung Cancer Detection System | Developed with ❤️")
