import streamlit as st
import requests

API_KEY = "IDE_MEGBÍZHATÓ_API_KULCS"

def get_competitions():
    url = "https://api.football-data.org/v4/competitions"
    headers = {"X-Auth-Token": API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("competitions", [])
    st.error(f"Hiba: {r.status_code}")
    return []

st.title("⚽ Football Competitions")
comps = get_competitions()
for c in comps:
    st.write(f"- **{c['name']}** ({c['area']['name']})")
