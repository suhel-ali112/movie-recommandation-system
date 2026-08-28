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

# CONFIG
st.set_page_config(
    page_title="PopcornAI",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750/18181C/A8A8A8?text=No+Poster"
OMDB_API_KEY = st.secrets.get("OMDB_API_KEY", "")
OMDB_BASE_URL = "http://www.omdbapi.com/"

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})
OMDB_TIMEOUT = 8

init_db()


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

# LOAD YOUR EXISTING ML ARTIFACTS — unchanged
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

# OMDb fetch — one call per movie TITLE (poster, overview, genres, rating,

@st.cache_data(show_spinner=False, ttl=60)
def fetch_movie_data(title):
    cached = get_cached_movie(title)
    if cached:
        return cached

    fallback = {
        "poster": PLACEHOLDER_POSTER, "overview": "No overview available.", "genres": "Unknown",
        "rating": 0, "runtime": "N/A", "release_date": "N/A", "rated": "NR", "imdb_id": "",
        "error": "",
    }

    if not OMDB_API_KEY:
        fallback["error"] = "missing_key"
        return fallback

    try:
        response = _session.get(
            OMDB_BASE_URL, params={"t": title, "apikey": OMDB_API_KEY}, timeout=OMDB_TIMEOUT
        )

        if response.status_code == 401:
            fallback["error"] = "invalid_key"
            return fallback

        response.raise_for_status()
        data = response.json()

        if data.get("Response") != "True":
            fallback["error"] = "not_found"
            return fallback

        def clean(field, default):
            val = data.get(field)
            return val if val and val != "N/A" else default

        poster = clean("Poster", "")
        imdb_rating = clean("imdbRating", None)
        result = {
            "poster": poster if poster else PLACEHOLDER_POSTER,
            "overview": clean("Plot", fallback["overview"]),
            "genres": clean("Genre", fallback["genres"]),
            "rating": float(imdb_rating) if imdb_rating else 0,
            "runtime": clean("Runtime", fallback["runtime"]),
            "release_date": clean("Released", clean("Year", fallback["release_date"])),
            "rated": clean("Rated", fallback["rated"]),
            "imdb_id": data.get("imdbID") or "",
            "error": "",
        }
        cache_movie(title, result) 
        return result
    except requests.exceptions.Timeout:
        fallback["error"] = "timeout"
        return fallback
    except Exception:
        fallback["error"] = "network"
        return fallback


OMDB_ERROR_MESSAGES = {
    "missing_key": "⚠️ OMDB_API_KEY is not set. Get a free key at omdbapi.com/apikey.aspx and add it to `.streamlit/secrets.toml` as `OMDB_API_KEY = \"your_key_here\"`.",
    "invalid_key": "⚠️ OMDb rejected the API key. Check that OMDB_API_KEY in secrets.toml is correct — and check your email to activate a newly-issued key.",
    "not_found": "🔍 OMDb doesn't have this title in its database.",
    "timeout": "⏳ OMDb requests are timing out — could be a slow connection. Try again in a bit.",
    "network": "⚠️ Couldn't reach OMDb — check your internet connection.",
}


def recommend(movie):
    """Your original cosine-similarity selection — unchanged. Posters are
    now fetched by title via OMDb, in parallel."""
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    recommended_movies = [movies.iloc[i[0]].title for i in movies_list]

    with ThreadPoolExecutor(max_workers=5) as executor:
        data_list = list(executor.map(fetch_movie_data, recommended_movies))
    recommended_movie_posters = [d["poster"] for d in data_list]

    return recommended_movies, recommended_movie_posters

# CURATED "Trending" / "Top Rated" lists — OMDb has no discovery endpoint,

TRENDING_TITLES = [
    "Dune: Part Two", "Oppenheimer", "Barbie", "Deadpool & Wolverine", "Inside Out 2",
    "Godzilla x Kong: The New Empire", "Furiosa: A Mad Max Saga", "The Fall Guy",
    "Bad Boys: Ride or Die", "Twisters", "Alien: Romulus", "Venom: The Last Dance",
    "Wicked", "Moana 2", "Gladiator II", "Sonic the Hedgehog 3",
    "Mufasa: The Lion King", "A Minecraft Movie", "Thunderbolts", "Superman",
]

TOP_RATED_TITLES = [
    "The Shawshank Redemption", "The Godfather", "The Dark Knight", "12 Angry Men",
    "Schindler's List", "The Lord of the Rings: The Return of the King", "Pulp Fiction",
    "The Good, the Bad and the Ugly", "Forrest Gump", "Fight Club", "Inception",
    "The Matrix", "Goodfellas", "Interstellar", "Parasite", "The Green Mile",
    "Se7en", "The Silence of the Lambs", "City of God", "Whiplash",
]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_curated_list(kind: str):
    titles = TRENDING_TITLES if kind == "trending" else TOP_RATED_TITLES
    with ThreadPoolExecutor(max_workers=5) as executor:
        data_list = list(executor.map(fetch_movie_data, titles))
    items = []
    for title, d in zip(titles, data_list):
        if d.get("error"):
            continue 
        items.append({"title": title, "poster": d["poster"], "rating": d["rating"]})
    return items

# LOGO — small clapperboard SVG mark + wordmark
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
# SESSION STATE
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("email", "")
st.session_state.setdefault("show_auth_modal", False) 
st.session_state.setdefault("page", "home")
st.session_state.setdefault("recommendations", None)
st.session_state.setdefault("loading", False)
st.session_state.setdefault("open_modal_for", None)
st.session_state.setdefault("watchlist", {})             

WATCHLIST_COUNT = len(st.session_state.watchlist)


def go_to(page_name: str):
    st.session_state.page = page_name
    st.session_state.recommendations = None
    st.session_state.loading = False
    st.session_state.open_modal_for = None   
    st.session_state.show_auth_modal = False   

def toggle_watchlist(title, poster):
    email = st.session_state.email
    if title in st.session_state.watchlist:
        remove_from_watchlist(email, title)
    else:
        add_to_watchlist(email, title, poster)
    st.session_state.watchlist = get_watchlist(email)


def logout():
    st.session_state.authenticated = False
    st.session_state.email = ""
    st.session_state.watchlist = {}
    st.session_state.page = "home"
    st.session_state.recommendations = None
    st.session_state.open_modal_for = None
    st.session_state.show_auth_modal = False

# DIALOG DISMISS HANDLERS

def _on_movie_modal_dismiss():
    st.session_state.open_modal_for = None


def _on_auth_modal_dismiss():
    st.session_state.show_auth_modal = False

# PROFILE MENU CONTENT — the "normal stuff" every account section needs
PRIVACY_POLICY_TEXT = (
    "We only store what's needed to run your account: your email, a securely "
    "hashed password (never stored in plain text), and your saved watchlist. "
    "We never sell or share your data with third parties. Movie posters and "
    "details are fetched from OMDb and are only cached briefly for speed."
)

TERMS_TEXT = (
    "PopcornAI is a student project built for learning purposes, not a "
    "commercial product. By creating an account you agree to use it "
    "responsibly. Recommendations are generated automatically by a machine "
    "learning model and may not always be accurate."
)

ABOUT_TEXT = (
    "PopcornAI uses a content-based recommendation model (TF-IDF / cosine "
    "similarity) trained on a movie dataset, paired with this Streamlit "
    "interface for browsing, searching, and saving movies. Poster art and "
    "details come from the OMDb API."
)

HELP_TEXT = (
    "Having trouble? Try refreshing the page, or check your internet "
    "connection — posters and details are fetched live from OMDb. For "
    "account issues, try logging out and logging back in."
)
# NAVBAR — logo | Home / Trending / Top Rated / Watchlist | 🔔 | profile
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
      
            if st.button("👤", key="open_auth_btn", help="Login / Sign Up"):
                st.session_state.show_auth_modal = True
                st.rerun()

@st.dialog(" ", on_dismiss=_on_auth_modal_dismiss)
def show_auth_modal():
    render_auth_page(LOGO_SVG)
    if st.button("Maybe later", key="auth_modal_close", use_container_width=True):
        st.session_state.show_auth_modal = False
        st.rerun()


if st.session_state.show_auth_modal and not st.session_state.authenticated:
    show_auth_modal()

# REUSABLE: render a grid of movie cards
def render_movie_grid(items, key_prefix, show_rating=True):
    cols = st.columns(5)
    for i, item in enumerate(items):
        with cols[i % 5]:
            title, poster = item["title"], item["poster"]
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
                    st.session_state.open_modal_for = (title, poster)
                    st.rerun()
            with btn_col2:
                is_fav = title in st.session_state.watchlist
                if st.button("❤️" if is_fav else "🤍", key=f"fav_{key_prefix}_{i}", use_container_width=True):
                    if not st.session_state.authenticated:
                        # guest tried to save — open login popup instead of failing silently
                        st.session_state.show_auth_modal = True
                    else:
                        toggle_watchlist(title, poster)
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


# PAGE: HOME

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

        
        if selected_movie:
            with st.spinner(""):
                preview_details = fetch_movie_data(selected_movie)
            if preview_details.get("error"):
                st.warning(OMDB_ERROR_MESSAGES.get(preview_details["error"], "OMDb fetch failed."))
            preview_poster = preview_details["poster"]
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
                st.session_state.open_modal_for = (selected_movie, preview_poster)
                st.rerun()

    if st.session_state.loading:
        st.markdown('<div class="cv-section-title">Finding movies for you</div>', unsafe_allow_html=True)
        render_skeleton(5)
        with st.spinner(""):
            names, posters = recommend(selected_movie)
            st.session_state.recommendations = [
                {"title": n, "poster": p} for n, p in zip(names, posters)
            ]
            st.session_state.loading = False
            st.rerun()

    if st.session_state.recommendations:
        st.markdown('<div class="cv-section-title">Recommended For You</div>', unsafe_allow_html=True)
        render_movie_grid(st.session_state.recommendations, key_prefix="home", show_rating=False)


# PAGE: TRENDING (curated list, refreshed hourly via OMDb)
elif st.session_state.page == "trending":
    st.markdown("""
    <div class="cv-page-banner"><h2>🔥 Trending Today</h2>
    <p>A curated set of popular titles, refreshed via OMDb.</p></div>
    """, unsafe_allow_html=True)
    with st.spinner("Loading trending movies..."):
        items = fetch_curated_list("trending")
    if items:
        render_movie_grid(items, key_prefix="trending", show_rating=True)
    else:
        render_empty("📡", "Couldn't reach OMDb", "Check your connection or API key and try again.")


# PAGE: TOP RATED (curated list, refreshed hourly via OMDb)
elif st.session_state.page == "top_rated":
    st.markdown("""
    <div class="cv-page-banner"><h2>🏆 Top Rated</h2>
    <p>Critically acclaimed classics, via OMDb.</p></div>
    """, unsafe_allow_html=True)
    with st.spinner("Loading top rated movies..."):
        items = fetch_curated_list("top_rated")
    if items:
        render_movie_grid(items, key_prefix="top_rated", show_rating=True)
    else:
        render_empty("📡", "Couldn't reach OMDb", "Check your connection or API key and try again.")

# PAGE: WATCHLIST — needs login; guests see a prompt instead of the grid

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
            {"title": t, "poster": d["poster"]}
            for t, d in st.session_state.watchlist.items()
        ]
        render_movie_grid(items, key_prefix="watchlist", show_rating=False)



@st.dialog(" ", on_dismiss=_on_movie_modal_dismiss)
def show_movie_modal(title, poster):
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"<img src='{poster}' style='width:100%;border-radius:14px;'/>", unsafe_allow_html=True)
    with c2:
       
        st.markdown(f"<div class='cv-modal-title'>{title}</div>", unsafe_allow_html=True)

        details_box = st.empty()
        with details_box.container():
            st.markdown("""
            <div class="cv-modal-skel-line" style="width:55%"></div>
            <div class="cv-modal-skel-line" style="width:35%"></div>
            <div class="cv-modal-skel-block"></div>
            """, unsafe_allow_html=True)

        details = fetch_movie_data(title)  

        with details_box.container():
            if details.get("error"):
                st.warning(OMDB_ERROR_MESSAGES.get(details["error"], "OMDb fetch failed."))
            st.markdown(
                f"<div class='cv-modal-meta'>{details['release_date']} • "
                f"{details['runtime']} • ⭐ {details['rating']}/10</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='cv-pill'>{details['genres']}</span>"
                f"<span class='cv-pill'>{details.get('rated', 'NR')}</span>",
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
                toggle_watchlist(title, poster)
            st.rerun()

        if st.button("Close", use_container_width=True, key="modal_close_btn"):
            st.session_state.open_modal_for = None
            st.rerun()


if st.session_state.open_modal_for:
    show_movie_modal(*st.session_state.open_modal_for)



# FOOTER

if st.session_state.authenticated:
    st.markdown(f"""
    <div class="cv-footer">
        Logged in as <b>{st.session_state.email}</b> &nbsp;•&nbsp;
        Built with ❤️ by team LastCommit &nbsp;•&nbsp; PopcornAI © 2026
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="cv-footer">
        Built with ❤️ by team LastCommit &nbsp;•&nbsp; PopcornAI © 2026
    </div>
    """, unsafe_allow_html=True)