"""Parse a match XML (UTF-16 LE: MatchSheet + Events) into a roster and an event table.

Usage:
    python tools/parse_events_to_df.py MATCH_EVENT.xml --players players.csv --events events.csv
"""
import argparse
import xml.etree.ElementTree as ET
import pandas as pd

def parse_match_data(xml_path):
    print(f"Loading and parsing {xml_path}...")
    with open(xml_path, 'r', encoding='utf-16-le') as f:
        xml_data = f.read()

    root = ET.fromstring(xml_data)

    # 1. Parse Match Sheet (Roster & Referees)
    match_sheet = root.find("MatchSheet")
    players_data = []
    for team in match_sheet.findall("Team"):
        team_name = team.get("Name")
        team_type = team.get("Type")
        team_id = team.get("IdTeam")
        for actor in team.findall("Actor"):
            first_name = actor.get("UsualFirstName") or ""
            nick_name = actor.get("NickName") or ""
            full_name = f"{first_name} {nick_name}".strip() if nick_name else first_name.strip()

            players_data.append({
                "Team": team_name,
                "TeamType": team_type,
                "TeamID": team_id,
                "Role": actor.get("Occupation"),
                "ActorID": actor.get("IdActor"),
                "Name": full_name,
                "Jersey": actor.get("JerseyNumber"),
                "Position": actor.get("Position"),
                "Starter": actor.get("IsStarter")
            })
    df_players = pd.DataFrame(players_data)

    # 2. Parse Events (Dynamic Attribute Extraction)
    events_element = root.find("Events")
    events_list = []
    for half in events_element.findall("EventsHalf"):
        half_id = half.get("IdHalf")
        for ev in half.findall("Event"):
            # Start with basic info
            ev_data = {
                "Half": int(half_id)
            }
            # Capture all XML attributes of this Event dynamically
            for attr, val in ev.attrib.items():
                ev_data[attr] = val

            events_list.append(ev_data)

    df_events = pd.DataFrame(events_list)

    # Clean up column types (convert numeric fields)
    numeric_cols = [
        "Time", "LocationX", "LocationY", "LocationZ",
        "TargetX", "TargetY", "TargetZ",
        "ScoreHomeTeam", "ScoreAwayTeam",
        "RedCardsHomeTeam", "RedCardsAwayTeam"
    ]
    for col in numeric_cols:
        if col in df_events.columns:
            df_events[col] = pd.to_numeric(df_events[col], errors='coerce')

    # Map player IDs to names for readability
    player_map = df_players.set_index("ActorID")["Name"].to_dict()
    if "IdActor1" in df_events.columns:
        df_events["Actor1Name"] = df_events["IdActor1"].map(player_map)
    if "IdActor2" in df_events.columns:
        df_events["Actor2Name"] = df_events["IdActor2"].map(player_map)
    if "DuelWinner" in df_events.columns:
        df_events["DuelWinnerName"] = df_events["DuelWinner"].map(player_map)

    print(f"Successfully parsed:")
    print(f"  - Players DataFrame: {df_players.shape[0]} rows, {df_players.shape[1]} columns")
    print(f"  - Events DataFrame: {df_events.shape[0]} rows, {df_events.shape[1]} columns")

    return df_players, df_events

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", help="match event XML (UTF-16 LE)")
    ap.add_argument("--players", help="write the roster to this CSV")
    ap.add_argument("--events", help="write the events to this CSV")
    a = ap.parse_args()

    df_players, df_events = parse_match_data(a.xml)
    if a.players:
        df_players.to_csv(a.players, index=False)
        print(f"Saved roster to: {a.players}")
    if a.events:
        df_events.to_csv(a.events, index=False)
        print(f"Saved events to: {a.events}")
