# 🌿 Patrick Smart Irrigation Dashboard v2.3

This version connects directly to **your own Google Drive sheet** for storage.

## 🚀 How to Use
1. Open your Google Sheet (e.g., `Patrick_Irrigation_Log`).
2. Share it with this email:  
   **patrick-irrigation-sa@patrick-irrigation-473904.iam.gserviceaccount.com**
3. Give it **Editor** access.
4. In your Streamlit app, keep your current service account JSON in secrets.

Optionally, to target a specific sheet, add this line in your secrets:
```toml
SHEET_ID = "your_google_sheet_id_here"
```
