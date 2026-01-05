import streamlit as st
from config import APP_TITLE


def apply_base_layout():
    # Global layout and styling for the app
    st.markdown(
        """
        <style>
        /* Hide Streamlit default menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Main container spacing - prevents headline from looking cut */
        .block-container {
            padding-top: 2.25rem;
            max-width: 1400px;
        }

        /* App headline styling */
        .app-headline {
            margin-top: 0.75rem;
            font-size: 2.6rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
            line-height: 1.15;
        }

        .app-subline {
            color: #475569;
            margin-bottom: 1.5rem;
            font-size: 1rem;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }

        /* Sticky tabs at top */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            position: sticky;
            top: 0;
            z-index: 999;
            background: white;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e5e7eb;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
            border-radius: 10px 10px 0 0;
            font-size: 0.95rem;
        }

        .stTabs [aria-selected="true"] {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_headline():
    # Main app title and subtitle
    st.markdown(
        f"<div class='app-headline'>{APP_TITLE}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-subline'>Upload, clean, analyze and visualize tabular data</div>",
        unsafe_allow_html=True,
    )