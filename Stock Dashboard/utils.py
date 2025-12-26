"""
Module containing utility functions for the financial analysis application.
"""

import streamlit as st
import pandas as pd

def config_menu_footer() -> None:
    """
    Customizes the Streamlit interface by hiding the default menu and footer,
    and adding a custom footer with copyright information.
    """
    app_style = """
        <style>
            /* Hide the default Streamlit menu */
            #MainMenu {
              visibility: hidden;
            }
            /* Hide the default Streamlit footer */
            footer {
                visibility: hidden;
            }
            /* Add a custom footer with centered text */
            footer:before {
                content:"Copyright © 2023 Abel Tavares";
                visibility: visible;
                display: block;
                position: relative;
                text-align: center;
            }
        </style>
    """
    st.markdown(app_style, unsafe_allow_html=True)

def get_delta(df: pd.DataFrame, key: str) -> str:
    """
    Computes the percentage change between the two most recent values for a specified key in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing financial metrics.
        key (str): The column name for which to calculate the percentage difference.

    Returns:
        str: The percentage difference formatted as a string with a '%' suffix.
    """
    if key not in df.columns:
        return f"Key '{key}' not found in DataFrame columns."

    if len(df) < 2:
        return "DataFrame must contain at least two rows."

    latest_value = df[key].iloc[0]
    previous_value = df[key].iloc[1]

    # Calculate percentage difference, handling negative or zero values appropriately
    if latest_value <= 0 or previous_value <= 0:
        delta = (previous_value - latest_value) / abs(latest_value) * 100
    else:
        delta = (previous_value - latest_value) / latest_value * 100

    return f"{delta:.2f}%"

def empty_lines(n: int) -> None:
    """
    Inserts a specified number of empty lines in the Streamlit app to create spacing.

    Parameters:
        n (int): The number of empty lines to insert.
    """
    for _ in range(n):
        st.write("")

def generate_card(text: str) -> None:
    """
    Creates a visually styled card with an icon and title text for display in the Streamlit app.

    Parameters:
        text (str): The title text to display on the card.
    """
    card_html = f"""
        <div style='border: 1px solid #e6e6e6; border-radius: 5px; padding: 10px; display: flex; justify-content: center; align-items: center'>
            <i class='fas fa-chart-line' style='font-size: 24px; color: #0072C6; margin-right: 10px'></i>
            <h3 style='text-align: center'>{text}</h3>
        </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def color_highlighter(val: str) -> str:
    """
    Applies color styling to DataFrame cells based on their value to enhance visual differentiation.

    Parameters:
        val (str): The cell value as a string.

    Returns:
        str: CSS style string to color the text red if negative; otherwise, no styling.
    """
    if val.startswith('-'):
        return 'color: rgba(255, 0, 0, 0.9);'  # Red color for negative values
    return None
