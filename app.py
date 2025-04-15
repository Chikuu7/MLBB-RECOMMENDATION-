import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
st.set_page_config(page_title="MLBB Hero Recommender", layout="wide")
st.title("🔥 MLBB Hero Recommendation and Visualization")
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("mlbb_heros.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["hero_name"] = df["hero_name"].str.strip().str.lower()
    df["role"] = df["role"].str.strip().str.lower()
    return df

df = load_data()
@st.cache_resource
def get_knn_model(data: pd.DataFrame):
    features = data[[
        "win_rate", "pick_rate", "offense_overall",
        "defense_overall", "skill_effect_overall", "difficulty_overall"
    ]]
    model = NearestNeighbors(n_neighbors=6, metric="euclidean")
    model.fit(features)
    return model

knn_model = get_knn_model(df)

# Sidebar Navigation
option = st.sidebar.radio("Choose Section", [
    "📄 Data Preview", "📍 Recommend by Lane", "🧠 Recommend Similar Heroes",
    "📊 Pick Rate Chart", "🆚 Compare Heroes", "🧩 Role Distribution", "📉 Stats Heatmap"
])

# 📄 Data Preview
if option == "📄 Data Preview":
    st.subheader("🗃 Hero Dataset Preview")

    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.slider("Number of rows to view", 5, 50, 10)
    with col2:
        show_cols = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())

    st.dataframe(df[show_cols].head(num_rows))

# 📍 Recommend by Lane
# 📍 Recommend by Lane (Role-based logic)
elif option == "📍 Recommend by Lane":
    st.subheader("🔍 Recommend Heroes by Lane")

    lane = st.selectbox("Select a lane", ["gold", "mid", "jungle", "roam", "exp"])
    lane = lane.strip().lower()

    role_map = {
        "gold": ["marksman"],
        "mid": ["mage", "support"],
        "roam": ["tank", "support"],
        "jungle": ["assassin", "fighter"],
        "exp": ["fighter", "tank"]
    }

    if lane in role_map:
        filtered = df[df["role"].isin(role_map[lane])]
        if not filtered.empty:
            st.dataframe(filtered[["hero_name", "role", "win_rate", "pick_rate"]].sort_values(by="win_rate", ascending=False))
        else:
            st.warning("No heroes found for that lane.")
    else:
        st.error("Invalid lane selected.")

elif option == "🧠 Recommend Similar Heroes":
    st.subheader("🤖 Similar Hero Recommendations using KNN")
    hero_input = st.text_input("Enter a hero name (e.g., martis)").strip().lower()

    if hero_input:
        if hero_input in df["hero_name"].values:
            hero_index = df[df["hero_name"] == hero_input].index[0]
            distances, indices = knn_model.kneighbors(
                df.loc[[hero_index], ["win_rate", "pick_rate", "offense_overall",
                                      "defense_overall", "skill_effect_overall", "difficulty_overall"]]
            )

            st.success(f"Top similar heroes to {hero_input.title()}:")
            for idx in indices[0][1:]:
                hero_row = df.iloc[idx]
                st.write(f"🧱 {hero_row['hero_name'].title()} — Role: {hero_row['role'].title()}, Win Rate: {hero_row['win_rate']}")
        else:
            st.error("Hero not found.")


# 📊 Pick Rate Chart
elif option == "📊 Pick Rate Chart":
    st.subheader("📈 Top 10 Heroes by Pick Rate")
    top = df.sort_values(by="pick_rate", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top["pick_rate"], y=top["hero_name"], ax=ax, palette="viridis")
    ax.set_xlabel("Pick Rate")
    ax.set_ylabel("Hero")
    st.pyplot(fig)

# 🆚 Compare Heroes
elif option == "🆚 Compare Heroes":
    st.subheader("📊 Compare Hero Stats")
    hero_names = st.text_input("Enter hero names (comma separated)", "martis, irithel, tigreal")
    names = [name.strip().lower() for name in hero_names.split(",")]
    compare = df[df["hero_name"].isin(names)]

    if not compare.empty:
        stats = ["win_rate", "pick_rate", "offense_overall", "defense_overall", "skill_effect_overall", "difficulty_overall"]
        melted = compare.melt(id_vars="hero_name", value_vars=stats, var_name="Stat", value_name="Value")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=melted, x="hero_name", y="Value", hue="Stat", ax=ax)
        ax.set_title("Hero Stats Comparison")
        st.pyplot(fig)
    else:
        st.warning("One or more heroes not found.")

# 🧩 Role Distribution
elif option == "🧩 Role Distribution":
    st.subheader("📊 Role Distribution")
    role_counts = df["role"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(role_counts, labels=role_counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# 📉 Stats Heatmap
elif option == "📉 Stats Heatmap":
    st.subheader("📉 Correlation Between Stats")
    stats = df[["win_rate", "pick_rate", "offense_overall", "defense_overall", "skill_effect_overall", "difficulty_overall"]]
    corr = stats.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
