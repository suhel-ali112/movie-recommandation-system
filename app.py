"""
PopcornAI — Discover Your Next Favorite Movie with AI
==========================================================
Frontend layer only. Your ML pipeline (TF-IDF/CountVectorizer + cosine
similarity, movies.pkl, similarity.pkl) is untouched.

THIS PASS
------------------------------------------------
1. Login/Signup are back to local SQLite (db.py) — no external OAuth
   setup needed. The only change: both forms ask for Gmail/Email
   instead of a username, and whatever email you sign up with is
   what you log in with.
2. Login/Signup popup only opens when the profile button (👤) is
   clicked, or when a guest tries to use the watchlist — not at
   startup.
3. 🔔 Notifications popover for app/movie updates.
4. "Get Recommendations" renamed to "Find" (same popcorn icon).
5. Modal-glitch fix, nav-wrap fix, colors/animations — unchanged.
"""

import pickle
import requests
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from db import (
    init_db, get_watchlist, add_to_watchlist, remove_from_watchlist,
    get_notifications, clear_notifications, get_cached_movie, cache_movie,
)
from auth import render_auth_page

# ==========================================================================
# CONFIG
# ==========================================================================
st.set_page_config(
    page_title="PopcornAI",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MOVIE_ID_COL = "movie_id"          # change to "id" if that's your column name
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750/18181C/A8A8A8?text=No+Poster"
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})
TMDB_TIMEOUT = 3

init_db()  # creates popcornai.db + tables on first run, no-op after that


def load_css(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


def time_ago(iso_str):
    """Turns a stored UTC timestamp into 'Xm ago' / 'Xh ago' / 'Xd ago'
    so notification times actually update as time passes (feels live)."""
    try:
        dt = datetime.fromisoformat(iso_str)
    except (TypeError, ValueError):
        return ""
    seconds = (datetime.utcnow() - dt).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


load_css("style.css")


# ==========================================================================
# LOAD YOUR EXISTING ML ARTIFACTS — unchanged
# ==========================================================================
@st.cache_resource
def load_data():
    try:
        movies = pickle.load(open("movies.pkl", "rb"))
        similarity = pickle.load(open("similarity.pkl", "rb"))
        return movies, similarity
    except FileNotFoundError:
        st.error(
            "⚠️ movies.pkl / similarity.pkl not found in the app directory. "
            "Place both files next to app.py (same folder)."
        )
        st.stop()


movies, similarity = load_data()


# ==========================================================================
# YOUR EXISTING BACKEND FUNCTION — logic unchanged, faster networking only
# ==========================================================================
POSTER_BASE = "https://image.tmdb.org/t/p/"


def poster_url(poster_path, size="w342"):
    """Builds a TMDB image URL from a raw poster_path at whatever size is
    needed. Grids use a smaller size (faster to download); the detail
    modal uses a bigger one — both come from the SAME cached fetch."""
    if not poster_path:
        return PLACEHOLDER_POSTER
    return f"{POSTER_BASE}{size}{poster_path}"


@st.cache_data(show_spinner=False)
def fetch_movie_data(movie_id):
    """One TMDB call per movie (poster + overview + genres + rating +
    runtime all come from the same /movie/{id} response — previously this
    was two separate calls). Cached twice over: st.cache_data keeps it
    instant for the rest of this session, and db.py persists it to
    SQLite so it's still instant after the app restarts, without
    re-hitting TMDB at all."""
    cached = get_cached_movie(movie_id)
    if cached:
        return cached

    fallback = {
        "poster_path": "", "overview": "No overview available.", "genres": "Unknown",
        "rating": 0, "runtime": "N/A", "release_date": "N/A", "popularity": "N/A",
    }
    try:
        url = (
            f"https://api.themoviedb.org/3/movie/{movie_id}"
            f"?api_key={TMDB_API_KEY}&language=en-US"
        )
        response = _session.get(url, timeout=TMDB_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        runtime = data.get("runtime")
        result = {
            "poster_path": data.get("poster_path") or "",
            "overview": data.get("overview") or fallback["overview"],
            "genres": ", ".join(g["name"] for g in data.get("genres", [])) or fallback["genres"],
            "rating": round(data.get("vote_average", 0), 1),
            "runtime": f"{runtime} min" if runtime else fallback["runtime"],
            "release_date": data.get("release_date") or fallback["release_date"],
            "popularity": f"{round(data.get('popularity', 0))}",
        }
        cache_movie(movie_id, result)  # persist to SQLite for future restarts
        return result
    except Exception:
        return fallback


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    recommended_movies = [movies.iloc[i[0]].title for i in movies_list]
    recommended_movie_ids = [movies.iloc[i[0]][MOVIE_ID_COL] for i in movies_list]

    with ThreadPoolExecutor(max_workers=5) as executor:
        data_list = list(executor.map(fetch_movie_data, recommended_movie_ids))
    recommended_movie_posters = [poster_url(d["poster_path"], "w342") for d in data_list]

    return recommended_movies, recommended_movie_posters, recommended_movie_ids


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tmdb_list(kind: str):
    url = (
        f"https://api.themoviedb.org/3/trending/movie/day?api_key={TMDB_API_KEY}"
        if kind == "trending"
        else f"https://api.themoviedb.org/3/movie/top_rated?api_key={TMDB_API_KEY}&language=en-US"
    )
    try:
        response = _session.get(url, timeout=TMDB_TIMEOUT)
        response.raise_for_status()
        results = response.json().get("results", [])[:20]
        items = []
        for m in results:
            poster_path = m.get("poster_path")
            items.append({
                "title": m.get("title") or m.get("name") or "Untitled",
                "movie_id": m.get("id"),
                "poster": poster_url(poster_path, "w342"),
                "rating": round(m.get("vote_average", 0), 1),
            })
        return items
    except Exception:
        return []


# ==========================================================================
# LOGO — small clapperboard SVG mark + wordmark
# ==========================================================================
LOGO_SVG = """
<svg class="cv-logo-mark" width="32" height="32" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="clapBody" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#232328"/>
      <stop offset="1" stop-color="#141417"/>
    </linearGradient>
    <linearGradient id="clapTop" x1="0" y1="0" x2="40" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#FF3B3B"/>
      <stop offset="1" stop-color="#E50914"/>
    </linearGradient>
  </defs>
  <rect x="4" y="15" width="32" height="21" rx="4" fill="url(#clapBody)"/>
  <g transform="rotate(-9 20 12)">
    <rect x="3" y="6" width="34" height="10" rx="2.5" fill="url(#clapTop)"/>
    <rect x="7" y="6" width="4" height="10" fill="#FFF6E0" transform="skewX(-20)"/>
    <rect x="16" y="6" width="4" height="10" fill="#FFF6E0" transform="skewX(-20)"/>
    <rect x="25" y="6" width="4" height="10" fill="#FFF6E0" transform="skewX(-20)"/>
  </g>
  <circle cx="32" cy="9" r="3.2" fill="#FFF6E0"/>
  <path d="M32 7.3l0.55 1.13 1.25 0.18-0.9 0.88 0.21 1.24L32 9.1l-1.11 0.63 0.21-1.24-0.9-0.88 1.25-0.18z" fill="#E50914"/>
</svg>
"""

# ==========================================================================
# SESSION STATE
# ==========================================================================
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("email", "")
st.session_state.setdefault("show_auth_modal", False)   # controls the login/signup popup
st.session_state.setdefault("page", "home")
st.session_state.setdefault("recommendations", None)
st.session_state.setdefault("loading", False)
st.session_state.setdefault("open_modal_for", None)
st.session_state.setdefault("watchlist", {})             # empty for guests, loaded from DB after login

WATCHLIST_COUNT = len(st.session_state.watchlist)


def go_to(page_name: str):
    st.session_state.page = page_name
    st.session_state.recommendations = None
    st.session_state.loading = False
    st.session_state.open_modal_for = None    # prevents old movie-modal-reappears glitch
    st.session_state.show_auth_modal = False   # prevents the login popup reappearing the same way


def toggle_watchlist(title, poster, movie_id):
    email = st.session_state.email
    if title in st.session_state.watchlist:
        remove_from_watchlist(email, title)
    else:
        add_to_watchlist(email, title, poster, movie_id)
    st.session_state.watchlist = get_watchlist(email)


def logout():
    st.session_state.authenticated = False
    st.session_state.email = ""
    st.session_state.watchlist = {}
    st.session_state.page = "home"
    st.session_state.recommendations = None
    st.session_state.open_modal_for = None
    st.session_state.show_auth_modal = False


# ==========================================================================
# PROFILE MENU CONTENT — the "normal stuff" every account section needs
# ==========================================================================
PRIVACY_POLICY_TEXT = (
    "We only store what's needed to run your account: your email, a securely "
    "hashed password (never stored in plain text), and your saved watchlist. "
    "We never sell or share your data with third parties. Movie posters and "
    "details are fetched live from TMDB and are only cached briefly for speed."
)

TERMS_TEXT = (
    "PopcornAI is a student project built for learning purposes, not a "
    "commercial product. By creating an account you agree to use it "
    "responsibly. Recommendations are generated automatically by a machine "
    "learning model and may not always be accurate."
)

ABOUT_TEXT = (
    "PopcornAI uses a content-based recommendation model (TF-IDF / cosine "
    "similarity) trained on the TMDB 5000 Movies dataset, paired with this "
    "Streamlit interface for browsing, searching, and saving movies."
)

HELP_TEXT = (
    "Having trouble? Try refreshing the page, or check your internet "
    "connection — posters and details are fetched live from TMDB. For "
    "account issues, try logging out and logging back in."
)


# ==========================================================================
# NAVBAR — logo | Home / Trending / Top Rated / Watchlist | 🔔 | profile
# ==========================================================================
with st.container(key="cv_navbar_row"):
    logo_col, home_col, trend_col, top_col, wl_col, spacer_col, bell_col, avatar_col = st.columns(
        [2.4, 0.8, 1.1, 1.25, 1.55, 1.75, 0.7, 0.75]
    )
    with logo_col:
        st.markdown(
            f'<div class="cv-logo-wrap">{LOGO_SVG}<span class="cv-logo-text">PopcornAI</span></div>',
            unsafe_allow_html=True,
        )

    nav_items = [
        (home_col, "Home", "home"),
        (trend_col, "Trending", "trending"),
        (top_col, "Top Rated", "top_rated"),
    ]
    for col, label, page_key in nav_items:
        with col:
            key_prefix = "nav_active" if st.session_state.page == page_key else "nav"
            if st.button(label, key=f"{key_prefix}_{page_key}"):
                go_to(page_key)
                st.rerun()

    with wl_col:
        wl_label = f"🔖 Watchlist ({WATCHLIST_COUNT})" if WATCHLIST_COUNT else "🔖 Watchlist"
        key_prefix = "nav_active" if st.session_state.page == "watchlist" else "nav"
        if st.button(wl_label, key=f"{key_prefix}_watchlist"):
            go_to("watchlist")
            st.rerun()

    with bell_col:
        # notifications are DB-backed: for a logged-in user they're keyed
        # to their email (clearing persists across logout -> login since
        # it's a DB row, not session_state). Cached in session_state so we
        # don't hit SQLite on every single rerun — only when the logged-in
        # identity actually changes (login/logout) or Clear is pressed.
        notif_email = st.session_state.email if st.session_state.authenticated else None
        cache_key = notif_email or "__guest__"
        if st.session_state.get("notif_cache_key") != cache_key:
            if notif_email:
                st.session_state.notifications = get_notifications(notif_email)
            else:
                st.session_state.notifications = (
                    [] if st.session_state.get("guest_notifs_cleared") else get_notifications(None)
                )
            st.session_state.notif_cache_key = cache_key
        notifications = st.session_state.notifications

        bell_wrap_key = "bell_wrap_active" if notifications else "bell_wrap_idle"
        with st.container(key=bell_wrap_key):
            with st.popover("🔔", use_container_width=False):
                header_col, clear_col = st.columns([2, 1])
                with header_col:
                    st.markdown('<div class="notif-header">🔔 Updates</div>', unsafe_allow_html=True)
                with clear_col:
                    if notifications:
                        if st.button("Clear", key="clear_notif_btn", use_container_width=True):
                            if notif_email:
                                clear_notifications(notif_email)
                            else:
                                st.session_state.guest_notifs_cleared = True
                            st.session_state.notifications = []
                            st.rerun()

                if not notifications:
                    st.markdown('<div class="notif-empty">You\'re all caught up 🎉</div>', unsafe_allow_html=True)
                else:
                    for n in notifications:
                        st.markdown(f"""
                        <div class="notif-item">
                            <div class="notif-icon">{n['icon']}</div>
                            <div class="notif-body">
                                <div class="notif-title">{n['title']}</div>
                                <div class="notif-desc">{n['desc']}</div>
                                <div class="notif-time">{time_ago(n['created_at'])}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    with avatar_col:
        if st.session_state.authenticated:
            initial = st.session_state.email[0].upper() if st.session_state.email else "?"
            with st.container(key="profile_wrap"):
                with st.popover(initial, use_container_width=False, help=st.session_state.email):
                    st.markdown(f'<div class="profile-email">{st.session_state.email}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="profile-sub">{WATCHLIST_COUNT} movie(s) in watchlist</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="profile-divider"></div>', unsafe_allow_html=True)
                    with st.expander("🔒 Privacy Policy"):
                        st.markdown(PRIVACY_POLICY_TEXT)
                    with st.expander("📜 Terms of Service"):
                        st.markdown(TERMS_TEXT)
                    with st.expander("ℹ️ About PopcornAI"):
                        st.markdown(ABOUT_TEXT)
                    with st.expander("❓ Help & Support"):
                        st.markdown(HELP_TEXT)
                    st.markdown('<div class="profile-divider"></div>', unsafe_allow_html=True)
                    if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                        logout()
                        st.rerun()
        else:
            # THIS is the profile button — login/signup ONLY opens when clicked here
            if st.button("👤", key="open_auth_btn", help="Login / Sign Up"):
                st.session_state.show_auth_modal = True
                st.rerun()


# ==========================================================================
# LOGIN / SIGNUP POPUP — only rendered when the profile button was clicked
# ==========================================================================
@st.dialog(" ")
def show_auth_modal():
    render_auth_page(LOGO_SVG)
    if st.button("Maybe later", key="auth_modal_close", use_container_width=True):
        st.session_state.show_auth_modal = False
        st.rerun()


if st.session_state.show_auth_modal and not st.session_state.authenticated:
    show_auth_modal()


# ==========================================================================
# REUSABLE: render a grid of movie cards
# ==========================================================================
def render_movie_grid(items, key_prefix, show_rating=True):
    cols = st.columns(5)
    for i, item in enumerate(items):
        with cols[i % 5]:
            title, poster, movie_id = item["title"], item["poster"], item["movie_id"]
            rating_html = (
                f'<div class="cv-rating-badge">⭐ {item.get("rating", "N/A")}</div>'
                if show_rating else ""
            )
            st.markdown(f"""
            <div class="cv-card">
                <div class="cv-poster-wrap">
                    <img src="{poster}" />
                    <div class="cv-poster-overlay"></div>
                    {rating_html}
                </div>
                <div class="cv-card-body">
                    <p class="cv-card-title">{title}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="cv-card-btn-row">', unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                if st.button("Details", key=f"details_{key_prefix}_{i}", use_container_width=True):
                    st.session_state.open_modal_for = (title, poster, movie_id)
                    st.rerun()
            with btn_col2:
                is_fav = title in st.session_state.watchlist
                if st.button("❤️" if is_fav else "🤍", key=f"fav_{key_prefix}_{i}", use_container_width=True):
                    if not st.session_state.authenticated:
                        # guest tried to save — open login popup instead of failing silently
                        st.session_state.show_auth_modal = True
                    else:
                        toggle_watchlist(title, poster, movie_id)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_skeleton(n=5):
    cols = st.columns(n)
    for c in cols:
        with c:
            st.markdown("""
            <div class="cv-skel">
                <div class="cv-skel-poster"></div>
                <div class="cv-skel-line" style="width:70%"></div>
            </div>
            """, unsafe_allow_html=True)


def render_empty(icon, title, subtitle):
    st.markdown(f"""
    <div class="cv-empty">
        <div class="cv-empty-icon">{icon}</div>
        <h3>{title}</h3>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================================
# PAGE: HOME
# ==========================================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="cv-hero">
        <h1>Popcorn<span>AI</span></h1>
        <p>Discover Your Next Favorite Movie with AI</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        select_col, btn_col = st.columns([3, 1])
        with select_col:
            selected_movie = st.selectbox(
                "movie", movies["title"].values,
                index=None,
                label_visibility="collapsed", placeholder="Type a movie name...",
            )
        with btn_col:
            if st.button("🍿 Find", key="cta_recommend", use_container_width=True) and selected_movie:
                st.session_state.loading = True
                st.session_state.recommendations = None

        # Instant preview of whatever is selected — fetched from TMDB the
        # moment the dropdown changes, before "Find" is even clicked.
        if selected_movie:
            preview_movie_id = movies.loc[movies["title"] == selected_movie, MOVIE_ID_COL].values[0]
            with st.spinner(""):
                preview_details = fetch_movie_data(preview_movie_id)
            preview_poster = poster_url(preview_details["poster_path"], "w342")
            st.markdown(f"""
            <div class="cv-preview-card">
                <img src="{preview_poster}" class="cv-preview-poster"/>
                <div class="cv-preview-info">
                    <div class="cv-preview-title">{selected_movie}</div>
                    <div class="cv-preview-meta">{preview_details['release_date']} • ⭐ {preview_details['rating']}/10</div>
                    <div class="cv-preview-genres">{preview_details['genres']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Details", key="details_preview", use_container_width=True):
                st.session_state.open_modal_for = (selected_movie, preview_poster, preview_movie_id)
                st.rerun()

    if st.session_state.loading:
        st.markdown('<div class="cv-section-title">Finding movies for you</div>', unsafe_allow_html=True)
        render_skeleton(5)
        with st.spinner(""):
            names, posters, ids = recommend(selected_movie)
            st.session_state.recommendations = [
                {"title": n, "poster": p, "movie_id": i} for n, p, i in zip(names, posters, ids)
            ]
            st.session_state.loading = False
            st.rerun()

    if st.session_state.recommendations:
        st.markdown('<div class="cv-section-title">Recommended For You</div>', unsafe_allow_html=True)
        render_movie_grid(st.session_state.recommendations, key_prefix="home", show_rating=False)

# ==========================================================================
# PAGE: TRENDING
# ==========================================================================
elif st.session_state.page == "trending":
    st.markdown("""
    <div class="cv-page-banner"><h2>🔥 Trending Today</h2>
    <p>What everyone's watching on TMDB right now.</p></div>
    """, unsafe_allow_html=True)
    with st.spinner("Loading trending movies..."):
        items = fetch_tmdb_list("trending")
    if items:
        render_movie_grid(items, key_prefix="trending", show_rating=True)
    else:
        render_empty("📡", "Couldn't reach TMDB", "Check your connection or API key and try again.")

# ==========================================================================
# PAGE: TOP RATED
# ==========================================================================
elif st.session_state.page == "top_rated":
    st.markdown("""
    <div class="cv-page-banner"><h2>🏆 Top Rated</h2>
    <p>The highest-rated titles on TMDB.</p></div>
    """, unsafe_allow_html=True)
    with st.spinner("Loading top rated movies..."):
        items = fetch_tmdb_list("top_rated")
    if items:
        render_movie_grid(items, key_prefix="top_rated", show_rating=True)
    else:
        render_empty("📡", "Couldn't reach TMDB", "Check your connection or API key and try again.")

# ==========================================================================
# PAGE: WATCHLIST — needs login; guests see a prompt instead of the grid
# ==========================================================================
elif st.session_state.page == "watchlist":
    st.markdown("""
    <div class="cv-page-banner"><h2>🔖 My Watchlist</h2>
    <p>Movies you've saved for later.</p></div>
    """, unsafe_allow_html=True)

    if not st.session_state.authenticated:
        render_empty("🔒", "Login required",
                     "Please log in or sign up first to save your watchlist.")
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            if st.button("👤 Login / Sign Up", key="watchlist_login_btn", use_container_width=True):
                st.session_state.show_auth_modal = True
                st.rerun()
    elif not st.session_state.watchlist:
        render_empty("🍿", "Your watchlist is empty",
                     "Tap the 🤍 icon on any movie card to save it here.")
    else:
        items = [
            {"title": t, "poster": d["poster"], "movie_id": d["movie_id"]}
            for t, d in st.session_state.watchlist.items()
        ]
        render_movie_grid(items, key_prefix="watchlist", show_rating=False)


# ==========================================================================
# DETAIL MODAL — native st.dialog, blurred backdrop
# ==========================================================================
@st.dialog(" ")
def show_movie_modal(title, poster, movie_id):
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"<img src='{poster}' style='width:100%;border-radius:14px;'/>", unsafe_allow_html=True)
    with c2:
        # Poster + title render immediately — they're already known from the
        # button click, so they're ALWAYS correct for the movie you just
        # opened. No more previous movie flashing before the new one loads.
        st.markdown(f"<div class='cv-modal-title'>{title}</div>", unsafe_allow_html=True)

        details_box = st.empty()
        with details_box.container():
            st.markdown("""
            <div class="cv-modal-skel-line" style="width:55%"></div>
            <div class="cv-modal-skel-line" style="width:35%"></div>
            <div class="cv-modal-skel-block"></div>
            """, unsafe_allow_html=True)

        details = fetch_movie_data(movie_id)  # cached (memory + DB) — instant on repeat opens, even after restart

        with details_box.container():
            st.markdown(
                f"<div class='cv-modal-meta'>{details['release_date']} • "
                f"{details['runtime']} • ⭐ {details['rating']}/10</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='cv-pill'>{details['genres']}</span>"
                f"<span class='cv-pill'>Popularity {details['popularity']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<p class='cv-modal-overview'>{details['overview']}</p>", unsafe_allow_html=True)

        already = title in st.session_state.watchlist
        label = "✓ In Watchlist — remove" if already else "➕ Add to Watchlist"
        if st.button(label, use_container_width=True, key="modal_watchlist_btn"):
            if not st.session_state.authenticated:
                st.session_state.open_modal_for = None
                st.session_state.show_auth_modal = True
            else:
                toggle_watchlist(title, poster, movie_id)
            st.rerun()

        if st.button("Close", use_container_width=True, key="modal_close_btn"):
            st.session_state.open_modal_for = None
            st.rerun()


if st.session_state.open_modal_for:
    show_movie_modal(*st.session_state.open_modal_for)


# ==========================================================================
# FOOTER
# ==========================================================================
if st.session_state.authenticated:
    st.markdown(f"""
    <div class="cv-footer">
        Logged in as <b>{st.session_state.email}</b> &nbsp;•&nbsp;
          Built with ❤️ by team LastCommit &nbsp;•&nbsp; PopcornAI © 2026 @pvt ltd
            </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="cv-footer">
        Built with ❤️ by team LastCommit &nbsp;•&nbsp; PopcornAI © 2026 @pvt ltd
    </div>
    """, unsafe_allow_html=True)
