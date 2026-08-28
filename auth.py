import streamlit as st
from db import signup_user, login_user, get_watchlist


def render_auth_page(logo_svg: str):
    st.markdown(f"""
    <div class="auth-hero">
        <div class="cv-logo-wrap" style="justify-content:center;">{logo_svg}</div>
        <h1>Popcorn<span>AI</span></h1>
        <p>Login or create a new account — for personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["🔑 Login", "✨ Sign Up"])

    # ---------------- LOGIN ----------------
    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Gmail / Email", key="login_email", placeholder="Enter your gmail..")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                ok, msg = login_user(email, password)
                if ok:
                    clean_email = email.strip().lower()
                    st.session_state.authenticated = True
                    st.session_state.email = clean_email
                    st.session_state.watchlist = get_watchlist(clean_email)  # load saved watchlist from DB
                    st.session_state.show_auth_modal = False
                    st.rerun()
                else:
                    st.error(msg)

    # ---------------- SIGN UP ----------------
    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
            new_email = st.text_input("Gmail / Email", key="signup_email", placeholder="Enter your gmail..")
            new_password = st.text_input("Choose a Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                ok, msg = signup_user(new_email, new_password, confirm_password)
                if ok:
                    st.success(msg + " 👆 Now go to the Login tab.")
                else:
                    st.error(msg)