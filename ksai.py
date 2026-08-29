import streamlit as st
import pickle
import random
import numpy as np
import datetime
import json
import os
import subprocess
import requests
from sklearn.metrics import pairwise_distances
from camel_tools.utils.dediac import dediac_ar
from camel_tools.disambig.mle import MLEDisambiguator
from streamlit_extras.let_it_rain import rain
from google import genai
from pydantic import BaseModel, Field
import traceback


@st.cache_resource
def setup_camel_data():
    # Newer camel-tools versions require the -i/--install flag and a real
    # catalog package name (the old "camel_data light" shortcut without -i
    # is from an older CLI version and no longer works).
    # disambig-mle-calima-msa-r13 pulls in its dependency
    # (morphology-db-msa-r13) automatically, covering everything
    # MLEDisambiguator.pretrained() needs.
    camel_data_dir = os.path.expanduser("~/.camel_tools/data/disambig_mle/calima-msa-r13")
    if not os.path.exists(camel_data_dir):
        with st.spinner("Downloading CAMEL Tools data... This may take a few minutes on first run."):
            subprocess.run(
                ["camel_data", "-i", "disambig-mle-calima-msa-r13"],
                check=True
            )

setup_camel_data()

#1__________________________________________________________________________________________________________لعنة الألوان
st.set_page_config(page_title="كلمة السر", page_icon="🔒", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Aref+Ruqaa:wght@700&display=swap');
    .stApp {
    background-color: #1a1c23 !important; 
    }
    /* تنسيق النص والاتجاه العام */
    .main { direction: rtl; color: black; text-align: right; }
    div[data-baseweb="input"] { direction: rtl; }
    
    /* تنسيق العنوان الرئيسي بخط الرقعة */
    .main-title {
        font-family: 'Aref Ruqaa', serif !important;
        font-size: 5rem !important; 
        text-align: center !important;
        color: white !important;
        margin-bottom: 30px !important;
        margin-top: 0px !important;
        font-weight: 700 !important;
        direction: rtl !important;
        line-height: 1 !important;
        display: block !important;
        width: 100% !important;
        min-height: auto !important;
        overflow: visible !important; 
    }

    /* تنسيق النص الفرعي */
    .sub-title {
        text-align: center !important;
        color: white !important; 
        font-size: 1rem !important; 
        margin-top: 20px !important; 
        margin-bottom: -20px !important; 
        
        font-weight: bold;
        direction: rtl;
    }

    /* تنسيق صناديق نتائج التخمين */
    .guess-box {
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        direction: rtl
    }
    
    /* تنسيق صندوق الإدخال */
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        color: black;
        margin-top: 0px !important; 
        background-color: white !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important;
        padding: 12px;
        length: 19px !important;
        border-radius: 10px;
        border:black
        font-weight: bold;
        font-size: 1.5rem !important;
        text-align: right;
        
    }
    /* st.error أو st.toast */
    .stAlert {
        direction: rtl !important;
        text-align: right !important;
        font-weight: bold !important;
        font-size: 1.5rem !important;
    }
    /* Press Enter */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    /* 1. تغيير لون الإطار عند النقر (Focus) أو التحويم (Hover) */
    .stTextInput > div:focus-within {
        border-color: #000000 !important; 
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.2) !important; 
    }

    .stTextInput > div:hover {
        border-color: #000000 !important; 
    }
    /* نقل الشريط الجانبي إلى اليمين */
    [data-testid="stSidebar"] {
        right: 0;
        left: auto;
        direction: rtl;
        text-align: right;
    }

    /* تعديل مكان محتوى الصفحة ليتناسب مع وجود الشريط على اليمين */
    [data-testid="stSidebarCollapsedControl"] {
        right: 20px;
        left: auto;
    }

    /* توحيد اتجاه النصوص داخل الشريط الجانبي */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] [data-testid="stHeading"],
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        direction: rtl !important;
        text-align: right !important;
    }
    </style>
    """, unsafe_allow_html=True)

#2________________________________________________________________________________________________________حامل الادوات

arabic_stops =  [
    "من", "إلى", "عن", "على", "في", "مع", "بين", "حول", "منذ", "خلال",
    "حتى", "أو", "ثم", "هل", "ما", "ماذا", "لماذا", "من", "متى", "أين",
    "كيف", "كم", "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "هنا", "هناك",
    "أنا", "أنت", "هو", "هي", "نحن", "هم", "كان", "ليس", "إن", "أن",
    "التي", "الذي", "الذين", "كل", "بعض", "كلما", "إذا", "لو", "لا", 
    "ما", "لم", "لن"
]


@st.cache_resource
def load_game_assets():
    mle = MLEDisambiguator.pretrained()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(BASE_DIR, "fasttext_arabic_limited.pkl")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    limit = 70000
    raw_words = data['words'][:limit]
    raw_embeddings = data['embeddings'][:limit]
    
    filtered_words = []
    filtered_embeddings = []
    for i, word in enumerate(raw_words):
        if word not in arabic_stops:
            filtered_words.append(word)
            filtered_embeddings.append(raw_embeddings[i])
    
    word_to_idx = {word: i for i, word in enumerate(filtered_words)}
    
    return mle, filtered_words, filtered_embeddings, word_to_idx

mle, words, embeddings, word_to_idx = load_game_assets()


def get_base_word(word):
    word = dediac_ar(word)
    shared_info = mle.disambiguate([word])
    if shared_info and len(shared_info[0].analyses) > 0:
        lemma = shared_info[0].analyses[0].analysis['lex']
        return dediac_ar(lemma)
    return word

#3________________________________________________________________________________________________________العقل المدبر

# Persistent storage via Upstash Redis REST API, so the daily word and
# word-history survive reboots/redeploys instead of living on the
# container's local disk (which gets wiped every restart).
# Set these two values in .streamlit/secrets.toml locally, and in your
# Streamlit Cloud app's "Secrets" settings when deployed:
#   UPSTASH_REDIS_REST_URL = "https://....upstash.io"
#   UPSTASH_REDIS_REST_TOKEN = "...."
UPSTASH_URL = st.secrets.get("UPSTASH_REDIS_REST_URL", os.environ.get("UPSTASH_REDIS_REST_URL", ""))
UPSTASH_TOKEN = st.secrets.get("UPSTASH_REDIS_REST_TOKEN", os.environ.get("UPSTASH_REDIS_REST_TOKEN", ""))

def _redis_command(*args, timeout=10):
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        return None
    resp = requests.post(
        UPSTASH_URL,
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        json=list(args),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("result")

def get_daily_word_record():
    """Returns (date_string, word) or (None, None) if nothing stored / storage unavailable."""
    try:
        raw = _redis_command("GET", "kalimat_alsir:daily_word")
        if raw:
            record = json.loads(raw)
            return record.get("date"), record.get("word")
    except Exception:
        pass
    return None, None

def set_daily_word_record(today_date, word):
    try:
        _redis_command("SET", "kalimat_alsir:daily_word", json.dumps({"date": today_date, "word": word}))
    except Exception:
        pass

def get_history():
    try:
        result = _redis_command("LRANGE", "kalimat_alsir:history", 0, -1)
        return result or []
    except Exception:
        return []

def append_history(word):
    try:
        _redis_command("RPUSH", "kalimat_alsir:history", word)
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

secret_words_pool = [
     "منبه", "شاحن", "مجهر", "كمامة", "محفظة", "ستارة", 
    "خرامة", "مروحة", "مسمار", "مفك", "ميزان", "جرار", "رافعة", 
    "بطارية", "عدسة", "بوق", "صندوق", "كهف", "بركان", "نفق", "جسر", 
    "واحة", "شلال", "رصيف", "قفل", "خرطوم", "شمعة", "صابون", "وسادة", 
    "منشفة", "معجون", "فرشاة", "دبوس", "قنبلة", "صاروخ", "ملصق", "طابع", 
    "عملة", "بصمة", "درع", "تابوت", "منصة", "صافرة"
]

class SecretWordSchema(BaseModel):
    word: str = Field(
        description=(
            "A singular Arabic noun representing a common, tangible object, tool, or place. "
            "It MUST have a rich semantic network (e.g., 'سفينة' which connects to sea, captain, waves). "
            "Strictly NO proper nouns (human/country names), NO abstract/philosophical concepts, "
            "NO hypernyms, NO archaic words, and NO diacritics."
        )
    )

def get_or_create_validated_daily_word(filtered_words, word_set=None, debug=False):
    """
    word_set: pass a set(filtered_words) from the caller for O(1) membership checks
              instead of re-hashing a big list on every 'in' check.
    debug: if True, show every rejected candidate + why, in the sidebar.
    """
    if word_set is None:
        word_set = set(filtered_words)

    today_date = str(datetime.date.today())

    if not UPSTASH_URL or not UPSTASH_TOKEN:
        st.sidebar.warning("⚠️ لم يتم إعداد التخزين الدائم (Upstash) — سيتم اختيار كلمة عشوائية عند كل إعادة تشغيل.")

    saved_date, saved_word = get_daily_word_record()
    if saved_date == today_date and saved_word:
        return saved_word

    past_words = get_history()

    recent_words_string = ", ".join(past_words[-3:]) if past_words else "لا يوجد"

    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

    def fallback_choice():
        available_backups = [w for w in secret_words_pool if w not in past_words and w in word_set]
        fallback_word = random.choice(available_backups if available_backups else list(filtered_words))
        set_daily_word_record(today_date, fallback_word)
        append_history(fallback_word)
        return fallback_word

    if not api_key:
        st.sidebar.warning("⚠️ لا يوجد مفتاح GEMINI_API_KEY — استخدام كلمة احتياطية.")
        return fallback_choice()

    try:
        client = genai.Client(api_key=api_key)
        attempts = 0
        max_attempts = 30
        rejected = []

        system_instruction = (
            "أنت خبير لغوي ومصمم ألعاب محترف لـ لعبة تخمين كلمات (سياق). "
            "مهمتك اختيار كلمة سر يومية ممتازة، ومثيرة للتحدي، وممتعة للاعبين.\n\n"
            "القواعد الذهبية لاختيار الكلمة:\n"
            "1. تصنيف الكلمة: يجب أن تكون (شيء مادي ملموس، أو أداة، أو مكان واضح) فقط.\n"
            "2. غنية بالروابط الدلالية: اختر كلمة تحيط بها شبكة واسعة من الكلمات المرتبطة بها في الأذهان.\n"
            "3. فصحى معاصرة ومألوفة ومحددة.\n"
            "4. قيود صارمة: يمنع منعاً باتاً أسماء العلم، الدول، الجمع، الأفعال، والتشكيل.\n\n"
            f"قاعدة حظر التكرار: يمنع تماماً اختيار أي كلمة من الكلمات السابقة: [{recent_words_string}]"
        )

        # NOTE: verify this model name is currently valid for your google-genai
        # SDK version/account — an invalid model id will make every attempt
        # below fail and silently fall through to the fallback word.
        model_name = "gemini-3.6-flash"

        while attempts < max_attempts:
            attempts += 1
            response = client.models.generate_content(
                model=model_name,
                contents="أعطني كلمة سر ممتازة ومبتكرة (شيء أو أداة أو مكان) للعبة سياق.",
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=SecretWordSchema,
                ),
            )

            res_data = json.loads(response.text)
            ai_word = res_data.get("word", "").strip()
            in_vocab = ai_word in word_set
            is_repeat = ai_word in past_words

            if in_vocab and not is_repeat:
                set_daily_word_record(today_date, ai_word)
                append_history(ai_word)
                return ai_word
            else:
                reason = "مكررة" if is_repeat else "غير موجودة في قاعدة البيانات"
                rejected.append((ai_word, reason))

        # Loop exhausted without success -> fall back instead of
        # falling off the end of the function and returning None.
        if debug and rejected:
            st.sidebar.warning(f"تم رفض {len(rejected)} كلمة من الذكاء الاصطناعي، تم استخدام كلمة احتياطية.")
        return fallback_choice()

    except Exception as e:
        if debug:
            st.sidebar.error(f"خطأ أثناء توليد الكلمة: {e}")
            st.sidebar.text(traceback.format_exc())
        return fallback_choice()


def random_game():
    past_words = get_history()

    if not past_words:
        past_words = secret_words_pool
        
    st.session_state.secret_word = random.choice(past_words)
    st.session_state.guesses = []
    st.session_state.last_guess = None
    
    v_secret = embeddings[word_to_idx[st.session_state.secret_word]]
    dist_matrix = pairwise_distances(v_secret.reshape(1, -1), embeddings, metric='cosine').reshape(-1)
    idxs = dist_matrix.argsort()
    rank_map = np.empty_like(idxs)
    rank_map[idxs] = np.arange(len(idxs))
    st.session_state.rank_map = rank_map
    st.session_state.sorted_words_for_today = [words[i] for i in idxs]


start_date = datetime.date(2026, 3, 21)
today = datetime.date.today()
days_diff = (today - start_date).days

if 'game_number' not in st.session_state:
    st.session_state.game_number = days_diff + 1

if 'guesses' not in st.session_state:
    st.session_state.guesses = []

if 'secret_word' not in st.session_state:
    word_set = set(words)
    st.session_state.secret_word = get_or_create_validated_daily_word(words, word_set=word_set, debug=True)
    
    v_secret = embeddings[word_to_idx[st.session_state.secret_word]]
    dist_matrix = pairwise_distances(v_secret.reshape(1, -1), embeddings, metric='cosine').reshape(-1)
    idxs = dist_matrix.argsort()
    rank_map = np.empty_like(idxs)
    rank_map[idxs] = np.arange(len(idxs))
    st.session_state.rank_map = rank_map
    st.session_state.sorted_words_for_today = [words[i] for i in idxs]

#4__________________________________________________________________________________________________________نشر مضاء
# -------------جانبية
if 'show_help' not in st.session_state:
    st.session_state.show_help = False
with st.sidebar:
    st.markdown("<div style='direction: rtl; text-align: right; font-family: Tahoma;'>", unsafe_allow_html=True)
    st.header(" :material/more_vert: خيارات اللعبة")
    
    
    if st.button(":material/help:  طريقة اللعب", use_container_width=True):
        st.session_state.show_help = not st.session_state.show_help
        if st.session_state.show_help:
            st.markdown(f"""
            <div style="
                direction: rtl; 
                text-align: right; 
                background-color: #eeeeee; 
                color: #333333; 
                padding: 20px; 
                border-radius: 10px; 
                border: 1px solid #cccccc;
                margin-bottom: 20px;
                line-height: 1.6;
            ">
                <strong style="font-size: 1.2rem;">ℹ️ طريقة اللعب</strong><br>
                <div style="padding-right: 10px; margin-top: 10px;">
                    • خمن الكلمة السرية (المحاولات غير محدودة).<br>
                    • الكلمات مرتبة بالذكاء الاصطناعي حسب تشابهها مع الهدف.<br>
                    • رتبة الكلمة السرية هي رقم 1.
                </div>
            </div>
            """, unsafe_allow_html=True)
    

    if st.button(":material/lightbulb: مســـــــاعدة" , use_container_width=True):
        if st.session_state.guesses:
            best_so_far = min(g['rank'] for g in st.session_state.guesses)
            limit = 300 if best_so_far > 300 else best_so_far
            if limit > 2:
                found_new = False
                forbidden_prefixes = ('ال', 'وال', 'ب', 'لل', 'بال', "مال","م",'ف')
                forbidden_suffixes = ('ه',"ان","ين","ون", 'ها', "ا",'هم', 'كن', 'كما','وا',"ات",'ي')
                # FIX: build candidates as (real_rank, word) pairs restricted to
                # the top `limit` words FIRST, then filter by prefix/suffix/length.
                # Previously the prefix/suffix filter was applied to the whole
                # list before indexing, which shifted word positions so a
                # "hint" could end up with a real rank far past `limit`.
                candidates = [
                    (i + 1, w)
                    for i, w in enumerate(st.session_state.sorted_words_for_today[:limit])
                    if i + 1 >= 2
                    and not w.startswith(forbidden_prefixes)
                    and not w.endswith(forbidden_suffixes)
                    and 8 > len(w) > 2
                ]
                random.shuffle(candidates)
                for rank, hint_word in candidates[:10]:
                    cleaned_hint = get_base_word(hint_word)
                    if not any(g['base'] == cleaned_hint for g in st.session_state.guesses):
                        st.session_state.guesses.append({'word': hint_word, "base": cleaned_hint, 'rank': int(rank)})
                        st.session_state.guesses.sort(key=lambda x: x['rank'])
                        st.session_state.last_guess = {'word': hint_word, "base": cleaned_hint, 'rank': int(rank)}
                        found_new = True
                        st.rerun()
                        break
               
                if not found_new:
                    st.sidebar.warning("حاول التخمين بنفسك قليلاً، لقد استهلكت تلميحات هذا النطاق!")
            else:
                st.sidebar.info(" 🤭 أنت قريب جداً!شغل عقلك.")
        else:
            st.sidebar.warning("قم بالتخمين أولاً لكي أستطيع مساعدتك!🫤")
                    

    if st.button(":material/refresh: إعادة محاولة", use_container_width=True):
        st.session_state.guesses = []
        st.session_state.last_guess = None
        st.rerun()
    if st.button(":material/shuffle: لعبة عشوائية", use_container_width=True):
        random_game()
        st.rerun()
        st.session_state.game_number = "🎲 عشوائيات"
    
    
        
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------- الوجه
st.markdown('<p class="main-title">كَلِمَةُ السِّرّ</p>', unsafe_allow_html=True)
is_winner = any(g['rank'] == 1 for g in st.session_state.guesses)
if 'last_guess' not in st.session_state:
    st.session_state.last_guess = None
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
    
if not is_winner:
    spaces = "&nbsp;" * 170
    st.markdown(f'<p class="sub-title">عدد المحاولات : {len(st.session_state.guesses)} {spaces} {st.session_state.game_number} # </p>', unsafe_allow_html=True)
    user_input = st.text_input("", placeholder="اكتب تخمينك هنا", key=f"guess_input_{st.session_state.input_key}")
    
    if user_input:
        string = str(user_input.strip())
        clean_guess = get_base_word(string)
        
        if clean_guess in arabic_stops:
             st.error(f"👿 '{string}' كلمة مضللة للمعنى.") 
        elif clean_guess not in word_to_idx:
            st.error(f"❌ '{string}' غير موجودة في قاعدة بياناتنا.") 
        elif any(g['base'] == clean_guess for g in st.session_state.guesses ):
            st.error(f"😅 '{string}' كلمة متكررة.")
        else:
            g_idx = word_to_idx[clean_guess]
            if clean_guess in  [get_base_word(st.session_state.secret_word),st.session_state.secret_word] or string in  [get_base_word(st.session_state.secret_word),st.session_state.secret_word] :
                rank = 1
                st.session_state.guesses.append({'word': string, "base": clean_guess,'rank': int(rank)})
                st.session_state.guesses.sort(key=lambda x: x['rank'])
                st.session_state.last_guess = {'word': string, 'base': clean_guess ,'rank': int(rank)}
                st.session_state.input_key += 1
                st.rerun()
            else:
                rank = st.session_state.rank_map[g_idx] + 1
                st.session_state.guesses.append({'word': string, "base": clean_guess,'rank': int(rank)})
                st.session_state.guesses.sort(key=lambda x: x['rank'])
                st.session_state.last_guess = {'word': string, 'base': clean_guess ,'rank': int(rank)}
                st.session_state.input_key += 1
                st.rerun()

   
    if st.session_state.last_guess:
        lg = st.session_state.last_guess
        r = lg['rank']
        color = "#74c69d" if r < 50 else "#40916c" if r < 150 else "#2d6a4f" if r < 300 else "#f2cf6f" if r < 800 else "#e89a51" if r < 1500 else "#e65a3e" if r < 2500 else "#c93e49" if r < 4000 else "#bf0f2a" if r < 6000 else "#750113"
        st.markdown(f"""
            <div class="guess-box" style="background-color: {color}; border: 3px solid #000000; color: white;">
                <span style="font-size: 1.1rem; font-weight: bold;">{lg['word']}</span>
                <span> {lg['rank']}</span>
            </div>
            <hr style="margin: 10px 0; border: 0; border-top: 1px solid #ccc;">
        """, unsafe_allow_html=True)
    for g in st.session_state.guesses:
        r = g['rank']
        color = "#74c69d" if r < 50 else "#40916c" if r < 150 else "#2d6a4f" if r < 300 else "#f2cf6f" if r < 800 else "#e89a51" if r < 1500 else "#e65a3e" if r < 2500 else "#c93e49" if r < 4000 else "#bf0f2a" if r < 6000 else "#750113"
        st.markdown(f"""
            <div class="guess-box" style="background-color: {color}; color: white;">
                <span style="font-size: 1.1rem; font-weight: bold;">{g['word']}</span>
                <span> {g['rank']}</span>
            </div>
        """, unsafe_allow_html=True)

else:  
    st.balloons()
    rain(
        emoji="✅",
        font_size=30,
        falling_speed=10,
        animation_length="infinite"
    )
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Aref+Ruqaa:wght@700&display=swap');
        
        .winner-card {{
            background: linear-gradient(135deg, #2d6a4f 0%, #1a1c23 100%);
            padding: 30px;
            border-radius: 20px;
            border: 2px solid #74c69d;
            box-shadow: 0px 0px 30px rgba(116, 198, 157, 0.5);
            text-align: center;
            direction: rtl;
            margin: 0 auto;
            animation: glow 2s infinite alternate;
        }}

        @keyframes glow {{
            from {{ box-shadow: 0px 0px 20px rgba(116, 198, 157, 0.4); }}
            to {{ box-shadow: 0px 0px 50px rgba(116, 198, 157, 0.8); }}
        }}

        .winner-title {{
            font-family: 'Aref Ruqaa', serif;
            font-size: 6rem;
            color: white !important;
            margin-bottom: 0px;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }}

        .winner-stats {{
            font-size: 2rem;
            color: #ffffff;
            margin-top: 20px;
        }}

        .secret-word-highlight {{
            color:#74c69d;
            font-size: 2.5rem;
            font-family: 'Aref Ruqaa', serif !important;
        }}
    </style>

    <div class="winner-card">
        <h1 class="winner-title">مبروك!</h1>
        <p class="winner-stats">
            لقد فككت الشفرة ووجدت كلمة السر <br>
            <span class="secret-word-highlight"> {st.session_state.secret_word} </span>
        </p>
        <p class="winner-stats">
            تم الإنجاز بعد <b>{len(st.session_state.guesses)}</b> محاولات
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    r = 0
    for g in st.session_state.sorted_words_for_today[:10]:
        r = r+1
        color = "#74c69d" if r < 50 else "#40916c" if r < 150 else "#2d6a4f" if r < 300 else "#f2cf6f" if r < 800 else "#e89a51" if r < 1500 else "#e65a3e" if r < 2500 else "#c93e49" if r < 4000 else "#bf0f2a" if r < 6000 else "#750113"
        st.markdown(f"""
        <div class="guess-box" style="background-color: {color}; color: white;">
            <span style="font-size: 1.1rem; font-weight: bold;">{g}</span>
            <span> {r}</span>
        </div>
        """, unsafe_allow_html=True)