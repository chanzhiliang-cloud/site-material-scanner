from pydantic import BaseModel, Field
import random
import time
import json
import streamlit as st
from google import genai
from google.genai import types

# 1. Define Pydantic Model for Guaranteed Schema Output
class MaterialAnalysis(BaseModel):
    item_name: str = Field(description="Name or brand of the construction material")
    specifications: str = Field(description="Dimensions, grade, thickness, or material type")
    unit: str = Field(description="Procurement unit, e.g., Per Piece, Per Length, Per Bag")
    low_price_myr: float = Field(description="Low-end market price in Malaysian Ringgit (MYR)")
    high_price_myr: float = Field(description="High-end market price in Malaysian Ringgit (MYR)")

# 2. Resilient Multimodal API Call Function
def call_gemini_with_fallback(client, img, prompt):
    """
    Tries primary, secondary, and lite models with fast exponential backoff
    to prevent thread locking while maintaining request resilience.
    """
    # Updated 2026 Model Fallback Pipeline
    models_to_try = [
        "gemini-2.5-flash", 
        "gemini-2.0-flash", 
        "gemini-2.0-flash-lite"
    ]
    max_retries_per_model = 3

    for model_name in models_to_try:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[img, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MaterialAnalysis,
                        temperature=0.2
                    )
                )
                return response.text
            except Exception as api_err:
                err_str = str(api_err)
                is_transient = any(code in err_str for code in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "Overloaded"])
                
                if is_transient and attempt < max_retries_per_model - 1:
                    # Exponential backoff with jitter: ~2s, ~4s, ~8s
                    wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    st.warning(f"[{model_name}] Busy (503/429). Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries_per_model})")
                    time.sleep(wait_time)
                else:
                    if model_name != models_to_try[-1]:
                        st.info(f"Switching from `{model_name}` to fallback model...")
                    break  # Move to next model in pipeline

    raise RuntimeError("Google Gemini free-tier endpoints are undergoing extreme traffic. Please wait 10 seconds and try again, or check your API billing tier in Google AI Studio.")
