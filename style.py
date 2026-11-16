import streamlit as st

CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 55%, #7c3aed 100%);
    border-radius: 20px; padding: 36px 32px; margin-bottom: 28px; color: #ffffff;
    box-shadow: 0 12px 30px -10px rgba(79, 70, 229, 0.55);
}
.hero h1 { color: #ffffff !important; font-size: 2.3rem; font-weight: 800; margin: 0 0 6px 0; padding: 0; }
.hero p  { color: #e0e7ff; font-size: 1.08rem; margin: 0; }
.hero .icon { font-size: 2.6rem; margin-bottom: 4px; }

.card {
    background: rgba(127, 127, 127, 0.07);
    border: 1px solid rgba(127, 127, 127, 0.25);
    border-radius: 16px; padding: 22px; margin-bottom: 18px; min-height: 175px;
    transition: all 0.25s ease;
}
.card:hover {
    transform: translateY(-4px); border-color: #6366f1;
    box-shadow: 0 14px 28px -12px rgba(99, 102, 241, 0.45);
}
.card .c-icon  { font-size: 2rem; }
.card .c-title { font-size: 1.15rem; font-weight: 700; margin: 6px 0 6px 0; }
.card .c-text  { font-size: 0.93rem; opacity: 0.78; line-height: 1.45; }
.card .c-tag {
    display: inline-block; margin-top: 10px; padding: 2px 10px;
    border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    background: rgba(99, 102, 241, 0.15); color: #6366f1;
}

.result { border-radius: 16px; padding: 20px 24px; margin: 8px 0 14px 0; border-left: 7px solid; }
.result .r-title { font-size: 1.35rem; font-weight: 800; margin-bottom: 4px; }
.result .r-text  { font-size: 1rem; opacity: 0.9; }
.result.ok   { background: rgba(34, 197, 94, 0.12);  border-color: #22c55e; }
.result.warn { background: rgba(239, 68, 68, 0.12);  border-color: #ef4444; }
.result.info { background: rgba(59, 130, 246, 0.12); border-color: #3b82f6; }

div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 12px; font-weight: 700; padding: 0.6rem 1rem;
}
[data-testid="stMetric"] {
    background: rgba(127, 127, 127, 0.07);
    border: 1px solid rgba(127, 127, 127, 0.2);
    border-radius: 14px; padding: 12px 16px;
}
footer {visibility: hidden;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(icon, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="icon">{icon}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def card(icon, title, text, tag=""):
    tag_html = f'<span class="c-tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="card"><div class="c-icon">{icon}</div>'
        f'<div class="c-title">{title}</div><div class="c-text">{text}</div>{tag_html}</div>',
        unsafe_allow_html=True,
    )


def result_box(kind, title, text=""):
    """kind : 'ok' (vert), 'warn' (rouge) ou 'info' (bleu)."""
    st.markdown(
        f'<div class="result {kind}"><div class="r-title">{title}</div>'
        f'<div class="r-text">{text}</div></div>',
        unsafe_allow_html=True,
    )