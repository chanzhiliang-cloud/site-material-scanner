import streamlit as st
import sqlite3
import pandas as pd
from PIL import Image
import json
import os
import time
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. DATABASE SETUP (Local SQLite)
# -----------------------------------------------------------------------------
DB_FILE = "material_inventory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            item_name TEXT,
            specifications TEXT,
            unit TEXT,
            low_price_myr REAL,
            high_price_myr REAL,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_record(item_name, specs, unit, low_price, high_price, notes):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO inventory (item_name, specifications, unit, low_price_myr, high_price_myr, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (item_name, specs, unit, low_price, high_price, notes))
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM inventory ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def delete_record(record_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM inventory WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

init_db()

# -----------------------------------------------------------------------------
# 2. HELPER: ROBUST GEMINI CALL (Retry + Fallback Strategy)
# -----------------------------------------------------------------------------
def call_gemini_with_fallback(client, img, prompt, response_schema):
    """
    Executes image analysis with model fallback and exponential retry backoff.
    """
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-pro"]
    max_retries_per_model = 3

    for model_name in models_to_try:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[img, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                )
                return response.text
            except Exception as api_err:
                err_str = str(api_err)
                is_503 = "503" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str
                
                # Retry if temporary server overload
                if is_503 and attempt < max_retries_per_model - 1:
                    wait_time = 2 * (attempt + 1)
                    st.warning(f"[{model_name}] Server busy (503). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries_per_model})")
                    time.sleep(wait_time)
                else:
                    # Move to fallback model if retries fail on primary
                    if model_name != models_to_try[-1]:
                        st.info(f"Primary model ({model_name}) unavailable. Switching to fallback model...")
                    break

    raise RuntimeError("All configured Gemini models are currently busy or unavailable. Please try again in a few moments.")

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Site Material Scanner", page_icon="🏗️", layout="wide")

st.title("🏗️ On-Site Material Price Finder")
st.caption("Snap a site photo to extract material specs and local Malaysian market prices (RM).")

# API Key handling
api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not api_key and "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]

# -----------------------------------------------------------------------------
# 4. FIELD INPUT (Camera or File Upload)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📸 Scan Material", "📊 Local Inventory History"])

with tab1:
    st.subheader("Capture / Upload Material")
    
    input_type = st.radio("Choose Input Method:", ["Camera Capture", "File Upload"], horizontal=True)
    
    image_file = None
    if input_type == "Camera Capture":
        image_file = st.camera_input("Take a photo of the construction material")
    else:
        image_file = st.file_uploader("Upload material photo", type=["jpg", "jpeg", "png", "webp"])

    if image_file:
        img = Image.open(image_file)
        st.image(img, caption="Target Material", use_container_width=True)
        
        notes = st.text_input("Additional Notes / Context (Optional)", placeholder="e.g., Block B, 3rd Floor wastage")
        
        if st.button("🚀 Identify & Search Market Prices", type="primary"):
            if not api_key:
                st.error("Please enter your Gemini API Key in the sidebar to proceed.")
            else:
                with st.spinner("Analyzing image and looking up current Malaysian market rates..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        prompt = """
                        You are a professional construction quantity surveyor and procurement manager in Malaysia.
                        Analyze the uploaded image of construction materials/equipment.
                        
                        Identify:
                        1. Item Name (Specific brand/type if visible).
                        2. Detailed Specifications (Dimensions, thickness, grade, or material type).
                        3. Standard Procurement Unit (e.g., Per Piece, Per Length, Per Board, Per Meter, Per Bag).
                        4. Low Price Range in MYR (RM) based on current local Malaysian hardware/distributor prices.
                        5. High Price Range in MYR (RM) based on current local Malaysian retail prices.
                        
                        Return ONLY JSON matching the requested fields.
                        """

                        json_schema = {
                            "type": "OBJECT",
                            "properties": {
                                "item_name": {"type": "STRING"},
                                "specifications": {"type": "STRING"},
                                "unit": {"type": "STRING"},
                                "low_price_myr": {"type": "NUMBER"},
                                "high_price_myr": {"type": "NUMBER"}
                            },
                            "required": ["item_name", "specifications", "unit", "low_price_myr", "high_price_myr"]
                        }

                        # Execute with retries & model fallback
                        response_text = call_gemini_with_fallback(client, img, prompt, json_schema)
                        data = json.loads(response_text)

                        # Save automatically to DB
                        insert_record(
                            data["item_name"],
                            data["specifications"],
                            data["unit"],
                            data["low_price_myr"],
                            data["high_price_myr"],
                            notes
                        )

                        st.success("✅ Analysis Complete & Saved to Internal App Database!")
                        
                        # Display Results
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Material", data["item_name"])
                        col2.metric("Unit", data["unit"])
                        col3.metric("Price Range (MYR)", f"RM {data['low_price_myr']:.2f} - RM {data['high_price_myr']:.2f}")

                        st.info(f"**Specs:** {data['specifications']}")

                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")

# -----------------------------------------------------------------------------
# 5. IN-APP STORAGE & HISTORY TAB
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Stored Material Records")
    df_records = load_data()
    
    if df_records.empty:
        st.info("No materials saved in the database yet.")
    else:
        st.dataframe(df_records, use_container_width=True)

        st.divider()
        st.subheader("Manage Database Records")
        record_to_delete = st.number_input("Enter ID to Delete", min_value=1, step=1)
        if st.button("Delete Entry"):
            delete_record(record_to_delete)
            st.warning(f"Record #{record_to_delete} deleted.")
            st.rerun()
