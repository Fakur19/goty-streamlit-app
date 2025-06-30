import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="Game Review Analysis Dashboard")
# st.set_option('deprecation.showPyplotGlobalUse', False)


# --- DATA LOADING ---
@st.cache_data
def load_data_from_s3():
    """
    Loads the FULLY PRE-PROCESSED Parquet data file from a private AWS S3 bucket.
    This function assumes the text cleaning and aspect extraction has already been done offline.
    """
    try:
        bucket_name = "goty-sentiment-analysis"
        # IMPORTANT: We are now loading the new, pre-processed file
        file_name = "reviews_processed.parquet"

        s3_path = f"s3://{bucket_name}/{file_name}"

        # df = pd.read_parquet(
        #     s3_path,
        #     storage_options={
        #         "key": st.secrets["aws"]["aws_access_key_id"],
        #         "secret": st.secrets["aws"]["aws_secret_access_key"],
        #     },
        # )

        df = pd.read_parquet("data/reviews_processed.parquet")

        # The data is already clean, but we still need to ensure dtypes are correct for plotting
        df["timestamp_created"] = pd.to_datetime(df["timestamp_created"])

        return df

    except Exception as e:
        st.error(f"Error loading data from AWS S3: {e}")
        st.info(
            "Please ensure: 1) 'reviews_processed.parquet' is in your S3 bucket. 2) Bucket name and secrets are correct."
        )
        return None


# --- MAIN APP ---
st.title("Game of the Year Steam Reviews Sentiment Analysis Analysis")

# Load image
st.image("img/goty.png")
# Load data
df_processed = load_data_from_s3()

if df_processed is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Introduction",
            "BERT Model Evaluation",
            "Player Sentiment Analysis",
            "Dominant Aspect Analysis",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Game Filter")
    game_list_all = ["All Games"] + sorted(df_processed["game"].unique().tolist())
    selected_game = st.sidebar.selectbox(
        "Select a game to filter visuals", game_list_all
    )

    # Filter data based on selection
    if selected_game == "All Games":
        filtered_df = df_processed
    else:
        filtered_df = df_processed[df_processed["game"] == selected_game]

    st.sidebar.markdown("---")  # Adds a visual separator

    st.sidebar.header("About")

    st.sidebar.markdown(
        "🐙 **Link:** [GitHub Repository](https://github.com/Fakur19/goty-sentiment-analysis)"  # Replace with your actual GitHub link
    )

    # --- PAGE CONTENT ---

    if page == "Introduction":
        try:
            st.header("Introduction")
            st.markdown("""
            This Streamlit application provides an in-depth analysis of Steam reviews for several "Game of the Year" award-winning titles.
            The analysis leverages Natural Language Processing (NLP) to understand player sentiment and identify key aspects of each game.

            **Key Features:**
            - **BERT Model Evaluation:** Assesses the performance of a pre-trained BERT model for sentiment classification against Steam's "voted up" labels.
            - **Player Sentiment Analysis:** Visualizes the overall sentiment distribution and tracks sentiment trends over time for each game.
            - **Dominant Aspect Analysis:** Identifies and quantifies the most frequently discussed aspects (e.g., Graphics, Story, Combat) in game reviews.

            Use the navigation panel on the left to explore the different sections of the analysis. You can also filter the visualizations by a specific game.
            """)

            st.subheader("Data Overview")
            st.write(
                f"The dataset contains **{len(df_processed):,}** reviews after cleaning and processing."
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("Game distribution in the dataset:")
                st.dataframe(df_processed["game"].value_counts())

            with st.expander("Click to view the processed data"):
                st.dataframe(df_processed.head(100))

            with st.expander("Click to view the processed positive sentiment"):
                df_positive = df_processed[
                    (df_processed["sentiment"] == "Positive")
                    & (df_processed["review"].str.len() > 2500)
                ]
                columns_to_display = [
                    "game",
                    "review",
                    "processed_review",
                    "sentiment",
                    "voted_up",
                ]
                st.dataframe(df_positive[columns_to_display].sample(100))

            with st.expander("Click to view the processed negative sentiment"):
                df_negative = df_processed[
                    (df_processed["sentiment"] == "Negative")
                    & (df_processed["review"].str.len() > 2500)
                ]
                columns_to_display = [
                    "game",
                    "review",
                    "processed_review",
                    "sentiment",
                    "voted_up",
                ]
                st.dataframe(df_negative[columns_to_display].sample(100))

        except Exception as e:
            st.error(f"An error occurred on the Introduction page.")
            st.exception(e)

    elif page == "BERT Model Evaluation":
        try:
            st.header("🤖 BERT Model Performance Evaluation")
            st.markdown("""
            This section evaluates the performance of the sentiment analysis model (a pre-trained BERT model) by comparing its predictions ('Positive'/'Negative')
            against the ground truth from Steam reviews ('voted_up' or 'voted_down').
            """)

            y_true = filtered_df["voted_up"].astype(int)
            y_pred = filtered_df["sentiment"].map({"Positive": 1, "Negative": 0})

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Classification Report")
                report = classification_report(
                    y_true,
                    y_pred,
                    target_names=["Negative", "Positive"],
                    output_dict=True,
                )
                st.dataframe(pd.DataFrame(report).transpose())

            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Label", y="True Label"),
                    x=["Negative", "Positive"],
                    y=["Negative", "Positive"],
                    title=f"Confusion Matrix: {selected_game}",
                )
                fig_cm.update_layout(title_x=0.5, title_font_size=22)
                st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred on the BERT Model Evaluation page.")
            st.exception(e)

    elif page == "Player Sentiment Analysis":
        try:
            st.header("Player Sentiment Analysis")

            # --- Sentiment Proportion ---
            st.subheader("Overall Sentiment Proportions")
            st.markdown(
                "This chart shows the percentage of positive vs. negative reviews for each game."
            )

            # --- Manual Percentage Calculation ---
            # 1. Start with the original counts
            sentiment_counts = (
                filtered_df.groupby(["game", "sentiment"])
                .size()
                .reset_index(name="count")
            )

            # 2. Calculate the total reviews for each game
            total_per_game = sentiment_counts.groupby("game")["count"].transform("sum")

            # 3. Create a new 'percentage' column
            sentiment_counts["percentage"] = (
                sentiment_counts["count"] / total_per_game
            ) * 100

            # --- Plotly Code Using the new 'percentage' column ---
            fig_sentiment = px.bar(
                sentiment_counts,
                x="game",
                # Use the manually calculated 'percentage' column for the y-axis
                y="percentage",
                color="sentiment",
                # barmode must be 'stack' for this to work
                barmode="stack",
                title=f"Sentiment Proportions for {selected_game}",
                labels={"percentage": "Percentage of Reviews (%)", "game": "Game"},
                color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728"},
            )

            # Customize the hover data and text inside the bars
            fig_sentiment.update_traces(
                texttemplate="%{y:.1f}%",
                textposition="inside",
                hovertemplate="<b>Game</b>: %{x}<br><b>Sentiment</b>: %{fullData.name}<br><b>Percentage</b>: %{y:.1f}%<extra></extra>",
            )
            fig_sentiment.update_layout(
                title_x=0.5, yaxis_title="Percentage (%)", title_font_size=22
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

            # --- Sentiment Trend ---
            st.subheader("Sentiment Trend Over Time")
            st.markdown(
                "This chart visualizes the volume of positive and negative reviews per month."
            )
            if selected_game == "All Games":
                st.info(
                    "Please select a single game from the sidebar to view its sentiment trend."
                )
            else:
                trend_df = filtered_df.copy()
                trend_df["review_month_dt"] = (
                    trend_df["timestamp_created"].dt.to_period("M").dt.to_timestamp()
                )
                monthly_sentiment = (
                    trend_df.groupby(["review_month_dt", "sentiment"])
                    .size()
                    .reset_index(name="count")
                )
                monthly_sentiment["signed_count"] = monthly_sentiment.apply(
                    lambda row: row["count"]
                    if row["sentiment"] == "Positive"
                    else -row["count"],
                    axis=1,
                )

                if not monthly_sentiment.empty:
                    fig_trend = px.bar(
                        monthly_sentiment,
                        x="review_month_dt",
                        y="signed_count",
                        color="sentiment",
                        title=f"Monthly Sentiment Trend for {selected_game}",
                        labels={
                            "review_month_dt": "Month",
                            "signed_count": "Number of Reviews",
                        },
                        color_discrete_map={
                            "Positive": "#2ca02c",
                            "Negative": "#d62728",
                        },
                        custom_data=["count"],
                    )
                    fig_trend.update_traces(
                        hovertemplate="<b>Month</b>: %{x}<br><b>Sentiment</b>: %{fullData.name}<br><b>Reviews</b>: %{customdata[0]}"
                    )
                    fig_trend.update_layout(
                        title_x=0.5,
                        barmode="relative",
                        height=600,
                        yaxis_title="Reviews (Positive vs. Negative)",
                        title_font_size=22,
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning(f"No monthly trend data available for {selected_game}.")

            # --- WORD CLOUDS ---
            st.subheader("Most Common Words in Reviews")
            st.markdown(
                "Word clouds showing the most frequent words in positive and negative reviews for the selected game."
            )

            def generate_wordcloud(text, title):
                """Generates and displays a word cloud."""
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    colormap="viridis",
                    max_words=150,
                ).generate(text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Positive Reviews")
                positive_reviews = filtered_df[
                    (filtered_df["sentiment"] == "Positive")
                    & (filtered_df["processed_review"].notna())
                ]
                positive_text = " ".join(positive_reviews["processed_review"])
                if positive_text:
                    generate_wordcloud(positive_text, "Positive Reviews")
                else:
                    st.warning("No positive reviews to generate a word cloud.")

            with col2:
                st.markdown("#### Negative Reviews")
                negative_reviews = filtered_df[
                    (filtered_df["sentiment"] == "Negative")
                    & (filtered_df["processed_review"].notna())
                ]
                negative_text = " ".join(negative_reviews["processed_review"])
                if negative_text:
                    generate_wordcloud(negative_text, "Negative Reviews")
                else:
                    st.warning("No negative reviews to generate a word cloud.")

        except Exception as e:
            st.error(f"An error occurred on the Player Sentiment Analysis page.")
            st.exception(e)

    elif page == "Dominant Aspect Analysis":
        try:
            st.header("🔍 Dominant Aspect Analysis")
            st.markdown(
                "This section breaks down what aspects of the games players are talking about in their reviews."
            )

            aspect_counts = (
                filtered_df.explode("aspects")
                .groupby(["game", "aspects"])
                .size()
                .reset_index(name="count")
            )

            # --- Proportions Chart ---
            st.subheader("Proportion of Dominant Aspects Mentioned")
            if not aspect_counts.empty:
                # --- Manual Percentage Calculation ---
                # 1. Calculate the total mentions for each game
                total_mentions = aspect_counts.groupby("game")["count"].transform("sum")

                # 2. Create a new 'percentage' column
                aspect_counts["percentage"] = (
                    aspect_counts["count"] / total_mentions
                ) * 100

                # --- Plotly Code Using the new 'percentage' column ---
                fig_aspect = px.bar(
                    aspect_counts,
                    x="game",
                    # Use the manually calculated 'percentage' column
                    y="percentage",
                    color="aspects",
                    barmode="stack",
                    title=f"Dominant Aspect Proportions for {selected_game}",
                    labels={"percentage": "Percentage of Mentions (%)", "game": "Game"},
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )

                # Customize the hover data and text inside the bars
                fig_aspect.update_traces(
                    texttemplate="%{y:.1f}%",
                    textposition="inside",
                    hovertemplate="<b>Game</b>: %{x}<br><b>Aspect</b>: %{fullData.name}<br><b>Percentage</b>: %{y:.1f}%<extra></extra>",
                )
                fig_aspect.update_layout(
                    title_x=0.5,
                    height=800,
                    title_font_size=25,
                    yaxis_title="Percentage of Mentions (%)",
                    legend_title_text="Aspects",
                )

                # Display the interactive chart
                st.plotly_chart(fig_aspect, use_container_width=True)
            else:
                st.warning(f"No aspect data to display for {selected_game}.")

            # --- Heatmaps ---
            st.subheader("Aspect Heatmaps")
            st.markdown(
                "Heatmaps showing the raw count of aspect mentions, categorized by overall, positive, and negative sentiment."
            )

            tab1, tab2, tab3 = st.tabs(
                ["All Sentiments", "Positive Sentiment", "Negative Sentiment"]
            )

            with tab1:
                st.markdown("#### Overall Aspect Mentions")
                if not aspect_counts.empty:
                    # Data prep is the same: pivot the table
                    aspect_matrix = aspect_counts.pivot_table(
                        index="game", columns="aspects", values="count", fill_value=0
                    )
                    aspect_matrix.loc["TOTAL"] = aspect_matrix.sum()

                    # Create the interactive heatmap with plotly.express
                    fig_heatmap_all = px.imshow(
                        aspect_matrix,
                        text_auto=True,  # Automatically displays the values on the cells
                        aspect="auto",  # Allows the cells to be rectangular
                        color_continuous_scale="Blues",  # Sets the color scheme
                        title=f"Aspect Categories in Reviews ({selected_game})",
                    )
                    fig_heatmap_all.update_layout(
                        title_x=0.5, height=700, title_font_size=22
                    )
                    st.plotly_chart(fig_heatmap_all, use_container_width=True)
                else:
                    st.warning(
                        f"No data for 'All Sentiments' heatmap for {selected_game}."
                    )

            with tab2:
                st.markdown("#### Positive Sentiment Aspect Mentions")
                positive_df = filtered_df[filtered_df["sentiment"] == "Positive"]
                aspect_counts_pos = (
                    positive_df.explode("aspects")
                    .groupby(["game", "aspects"])
                    .size()
                    .reset_index(name="count")
                )

                if not aspect_counts_pos.empty:
                    aspect_matrix_pos = aspect_counts_pos.pivot_table(
                        index="game", columns="aspects", values="count", fill_value=0
                    )
                    aspect_matrix_pos.loc["TOTAL"] = aspect_matrix_pos.sum()

                    fig_heatmap_pos = px.imshow(
                        aspect_matrix_pos,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Greens",
                        title=f"Aspect Categories in Positive Reviews ({selected_game})",
                    )
                    fig_heatmap_pos.update_layout(
                        title_x=0.5, height=700, title_font_size=22
                    )
                    st.plotly_chart(fig_heatmap_pos, use_container_width=True)
                else:
                    st.warning(
                        f"No data for 'Positive Sentiment' heatmap for {selected_game}."
                    )

            with tab3:
                st.markdown("#### Negative Sentiment Aspect Mentions")
                negative_df = filtered_df[filtered_df["sentiment"] == "Negative"]
                aspect_counts_neg = (
                    negative_df.explode("aspects")
                    .groupby(["game", "aspects"])
                    .size()
                    .reset_index(name="count")
                )

                if not aspect_counts_neg.empty:
                    aspect_matrix_neg = aspect_counts_neg.pivot_table(
                        index="game", columns="aspects", values="count", fill_value=0
                    )
                    aspect_matrix_neg.loc["TOTAL"] = aspect_matrix_neg.sum()

                    fig_heatmap_neg = px.imshow(
                        aspect_matrix_neg,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Reds",
                        title=f"Aspect Categories in Negative Reviews ({selected_game})",
                    )
                    fig_heatmap_neg.update_layout(
                        title_x=0.5, height=700, title_font_size=22
                    )
                    st.plotly_chart(fig_heatmap_neg, use_container_width=True)
                else:
                    st.warning(
                        f"No data for 'Negative Sentiment' heatmap for {selected_game}."
                    )

        except Exception as e:
            st.error(f"An error occurred on the Dominant Aspect page.")
            st.exception(e)
else:
    st.error(
        "Application cannot start because the necessary data files could not be loaded. Please check the file paths and try again."
    )
