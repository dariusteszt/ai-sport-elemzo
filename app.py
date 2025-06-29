import streamlit as st
import requests

# IDE ÍRD BE A SAJÁT API-FOOTBALL KULCSOD
API_KEY = "IDE_ÍRD_BE"

def get_leagues():
    url = "https://v3.football.api-sports.io/leagues"
    headers = {
        "x-apisports-key": API_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("response", [])
    else:
        st.error("⚠️ Hiba történt az API hívás során.")
        return []

def main():
    st.set_page_config(page_title="⚽ API Football Ligák", layout="centered")
    st.title("⚽ Futball Ligák Listája - API Football")

    st.markdown("Ez az app az [api-football.com](https://www.api-football.com/) API-jából jeleníti meg a ligákat.")

    leagues = get_leagues()

    if leagues:
        for league_info in leagues[:20]:  # csak az első 20 ligát mutatjuk
            league = league_info.get("league", {})
            country = league_info.get("country", {})
            st.markdown(f"- **{league.get('name')}** ({country.get('name')})")
    else:
        st.warning("Nincs elérhető liga adat.")

if __name__ == "__main__":
    main()
