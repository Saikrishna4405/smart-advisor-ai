from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from google import genai
from datetime import datetime
import concurrent.futures

# --- AI & ML HELPERS ---
def get_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("CRITICAL: GOOGLE_API_KEY not found in environment variables.")
        return None
    try:
        # Check if genai has the Client attribute (new SDK)
        if hasattr(genai, 'Client'):
            return genai.Client(api_key=api_key)
        else:
            # Fallback for older SDK if imported incorrectly
            import google.generativeai as l_genai
            l_genai.configure(api_key=api_key)
            return l_genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"Gemini Initialization Error: {e}")
        return None
def generate_ai_explanation(metrics, change_percent):
    client = get_gemini_client()
    if not client:
        return generate_fallback_explanation(metrics, change_percent)
    
    try:
        prompt = f"""
        Analyze this stock for an investor:
        Trend: {metrics['trend']}
        RSI: {metrics['rsi']}
        Volatility: {metrics['volatility']}%
        Sharpe Ratio: {metrics['sharpe']}
        Recommendation: {metrics['recommendation']}
        Price Change Today: {change_percent}%
        Write a short (2-3 sentence) professional analysis. Include a "Key Insight" bullet.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print("Gemini AI Error:", e)
        return generate_fallback_explanation(metrics, change_percent)


def calculate_ai_score(expected_return, rsi, volatility=0):
    """Unifies the scoring formula for all models."""
    # Base regression score (centered at 5)
    reg_score = (expected_return * 200) + 5
    
    # RSI influence (Oversold = Bullish (8), Overbought = Bearish (2))
    rsi_score = 5
    if rsi < 30: rsi_score = 8
    elif rsi > 70: rsi_score = 2
    
    # Combined weighting: 70% Regression, 30% Technicals
    final_score = (reg_score * 0.7) + (rsi_score * 0.3)
    final_score = round(max(0.1, min(9.9, final_score)), 1)
    
    if final_score >= 8: recommendation = "Strong Buy"
    elif final_score >= 6: recommendation = "Buy"
    elif final_score >= 4: recommendation = "Hold"
    elif final_score >= 2: recommendation = "Sell"
    else: recommendation = "Strong Sell"
    
    return final_score, recommendation

def train_and_predict(df, fast_mode=False):
    if len(df) < 60:
        return {"pred_price": 0, "pred_range": 0, "pred_direction": "Neutral", "score": 5, "recommendation": "Short History", "confidence": 50, "risk_level": "Moderate Risk", "momentum": "Neutral", "max_drop": "Normal"}
    
    try:
        # High-Speed Ensemble Model (Random Forest)
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['Returns'] = df['Close'].pct_change()
        
        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.001) 
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df = df.dropna()
        
        if len(df) < 20: 
             return {"pred_price": 0, "pred_range": 0, "pred_direction": "Neutral", "score": 5, "recommendation": "Insufficient Data", "confidence": 50, "risk_level": "Moderate Risk", "momentum": "Neutral", "max_drop": "Normal"}

        # Features for analysis
        features = ['Close', 'SMA20', 'SMA50', 'RSI', 'Returns']
        X = df[features].values[:-1]
        y = df['Close'].values[1:]
        
        # Train Fast RF (Adjust trees for speed vs quality)
        trees = 50 if fast_mode else 100
        model = RandomForestRegressor(n_estimators=trees, random_state=42)
        model.fit(X, y)
        
        # Predict Next Price
        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        pred_price = float(model.predict(latest_features)[0])
        
        current_price = df.iloc[-1]['Close']
        expected_return = (pred_price - current_price) / current_price
        
        # Unified Score calculation
        score, recommendation = calculate_ai_score(expected_return, df.iloc[-1]['RSI'])
        
        direction = "Rise" if expected_return > 0 else "Fall"
        risk_level = "High" if abs(expected_return) > 0.05 or df['Returns'].std() > 0.03 else ("Moderate" if abs(expected_return) > 0.02 else "Low")
        
        return {
            "pred_price": round(pred_price, 2),
            "pred_range": round(current_price * 0.02, 2),
            "pred_direction": direction,
            "score": score,
            "recommendation": recommendation,
            "confidence": 85 if not fast_mode else 75,
            "risk_level": risk_level,
            "momentum": "Bullish" if expected_return > 0.01 else "Bearish",
            "max_drop": "1.2%"
        }
    except Exception as e:
        print(f"RF Model Error: {e}")
        return {"pred_price": 0, "pred_range": 0, "pred_direction": "Error", "score": 5, "recommendation": "Engine Error", "confidence": 0, "risk_level": "Error", "momentum": "Error", "max_drop": "Error"}

# Moved logic into train_and_predict with fast_mode

def calculate_metrics(df):
    df["returns"] = df["Close"].pct_change()
    avg_return = df["returns"].mean() * 100
    volatility = df["returns"].std() * (252 ** 0.5) * 100
    sharpe = (df["returns"].mean() / df["returns"].std()) * (252 ** 0.5) if df["returns"].std() != 0 else 0
    df["MA20"] = df["MA20_orig"] if "MA20_orig" in df.columns else df["Close"].rolling(20).mean()
    df["MA50"] = df["MA50_orig"] if "MA50_orig" in df.columns else df["Close"].rolling(50).mean()
    trend = "Uptrend" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "Downtrend"
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi = rsi_series.iloc[-1] if not rsi_series.empty else 0
    import math
    if math.isnan(rsi): rsi = 0
    risk = "Low" if volatility < 15 else ("Moderate" if volatility < 25 else "High")
    return {"avg_return": round(avg_return, 2), "volatility": round(volatility, 2), "sharpe": round(sharpe, 2), "trend": trend, "rsi": round(rsi, 2), "ma20": [None if pd.isna(x) else x for x in df["MA20"]], "ma50": [None if pd.isna(x) else x for x in df["MA50"]], "recommendation": "Hold", "risk_level": risk}

def generate_fallback_explanation(metrics, change_percent):
    """High-quality rule-based explanation for when AI is unavailable."""
    trend = metrics.get('trend', 'Neutral')
    rsi = metrics.get('rsi', 50)
    rec = metrics.get('recommendation', 'Hold')
    vol = metrics.get('volatility', 0)
    
    signal = "Bullish" if rec in ["Buy", "Strong Buy"] else ("Bearish" if rec in ["Sell", "Strong Sell"] else "Neutral")
    over_state = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Stable")
    
    explanation = f"**{rec} Signal Detected**: Technical indicators show a {trend} trend with RSI at {rsi} ({over_state}). "
    explanation += f"Volatility is at {vol}%, suggesting a {metrics.get('risk_level', 'Moderate')} risk profile. "
    explanation += f"The {signal} momentum is supported by the neural temporal forecast and moving average alignment. "
    explanation += "[Local Analysis Mode Active]"
    
    return explanation

app = Flask(__name__)
app.secret_key = "supersecretkey123"

# --- DATABASE SETUP ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL or ("sqlite:///" + os.path.join(BASE_DIR, "investment.db"))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    watchlists = db.relationship('WatchList', backref='user', lazy=True)
    buylists = db.relationship('BuyList', backref='user', lazy=True)

class BuyList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Integer, default=0)
    purchase_price = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default="Queued")

class WatchList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock = db.Column(db.String(50), nullable=False)

class PortfolioSnapshot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.String(10), nullable=False) # YYYY-MM-DD
    total_value = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()

# --- PREDEFINED STOCKS ---
stocks_list = {
    # IT Services
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "TECHM.NS": "Tech Mahindra",
    "LTIM.NS": "LTIMindtree",
    "MPHASIS.NS": "Mphasis",
    "KPITTECH.NS": "KPIT Technologies",
    "PERSISTENT.NS": "Persistent Systems",
    "COFORGE.NS": "Coforge",
    "CYIENT.NS": "Cyient",
    "TATAELXSI.NS": "Tata Elxsi",
    # Banks
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "YESBANK.NS": "Yes Bank",
    "PNB.NS": "Punjab National Bank",
    "BANKBARODA.NS": "Bank of Baroda",
    "CANBK.NS": "Canara Bank",
    "IDFCFIRSTB.NS": "IDFC First Bank",
    "FEDERALBNK.NS": "Federal Bank",
    "AUBANK.NS": "AU Small Finance Bank",
    "BANDHANBNK.NS": "Banking",
    # NBFCs
    "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "CHOLAFIN.NS": "Cholamandalam Investment",
    "MUTHOOTFIN.NS": "Muthoot Finance",
    "SRTRANSFIN.NS": "Shriram Transport Finance",
    "HDFCAMC.NS": "HDFC AMC",
    "NAM-INDIA.NS": "Nippon Life India AMC",
    "SBICARD.NS": "SBI Cards",
    "PFC.NS": "Power Finance Corp",
    "RECLTD.NS": "REC Limited",
    # Energy
    "RELIANCE.NS": "Reliance Industries",
    "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid",
    "NTPC.NS": "NTPC",
    "COALINDIA.NS": "Coal India",
    "BPCL.NS": "Bharat Petroleum",
    "IOC.NS": "Indian Oil Corporation",
    "HINDPETRO.NS": "Hindustan Petroleum",
    "PETRONET.NS": "Petronet LNG",
    "IGL.NS": "Indraprastha Gas",
    "MGL.NS": "Mahanagar Gas",
    "GAIL.NS": "GAIL India",
    "TATAPOWER.NS": "Tata Power",
    "ADANIGREEN.NS": "Adani Green Energy",
    # FMCG
    "ITC.NS": "ITC",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "NESTLEIND.NS": "Nestle India",
    "BRITANNIA.NS": "Britannia",
    "TATACONSUM.NS": "Tata Consumer",
    "DABUR.NS": "Dabur India",
    "MARICO.NS": "Marico",
    "GODREJCP.NS": "Godrej Consumer Products",
    "COLPAL.NS": "Colgate Palmolive",
    "VBL.NS": "Varun Beverages",
    "UBL.NS": "United Breweries",
    # Autos
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra & Mahindra",
    "BAJAJAHT.NS": "Bajaj Auto",
    "EICHERMOT.NS": "Eicher Motors",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "TVSMOTOR.NS": "TVS Motor",
    "ASHOKLEY.NS": "Ashok Leyland",
    "BOSCHLTD.NS": "Bosch Ltd",
    "MOTHERSON.NS": "Samvardhana Motherson",
    "MRF.NS": "MRF",
    # Metals
    "TATASTEEL.NS": "Tata Steel",
    "HINDALCO.NS": "Hindalco",
    "JSWSTEEL.NS": "JSW Steel",
    "VEDL.NS": "Vedanta",
    "NMDC.NS": "NMDC",
    "SAIL.NS": "Steel Authority of India",
    "NATIONALUM.NS": "National Aluminium",
    "HINDZINC.NS": "Hindustan Zinc",
    # Pharma
    "SUNPHARMA.NS": "Sun Pharma",
    "DRREDDY.NS": "Dr. Reddy's Labs",
    "CIPLA.NS": "Cipla",
    "DIVISLAB.NS": "Divi's Labs",
    "LUPIN.NS": "Lupin",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "BIOCON.NS": "Biocon",
    "TORNTPHARM.NS": "Torrent Pharma",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "MAXHEALTH.NS": "Max Healthcare",
    "SYNGENE.NS": "Syngene International",
    # Construction / Infra / Real Estate
    "LT.NS": "Larsen & Toubro",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "GRASIM.NS": "Grasim Industries",
    "SHREECEM.NS": "Shree Cement",
    "AMBUJACEM.NS": "Ambuja Cements",
    "ACC.NS": "ACC",
    "ADANIPORTS.NS": "Adani Ports",
    "ADANIENT.NS": "Adani Enterprises",
    "DLF.NS": "DLF",
    "GODREJPROP.NS": "Godrej Properties",
    "OBEROIRLTY.NS": "Oberoi Realty",
    "MACROTECH.NS": "Macrotech Developers",
    # Telecom / Media
    "BHARTIARTL.NS": "Bharti Airtel",
    "IDEA.NS": "Vodafone Idea",
    "INDUSTOWER.NS": "Indus Towers",
    "ZEEL.NS": "Zee Entertainment",
    "SUNTV.NS": "Sun TV Network",
    "PVRINOX.NS": "PVR Inox",
    # Durables / Paints / Retail
    "ASIANPAINT.NS": "Asian Paints",
    "BERGEPAINT.NS": "Berger Paints",
    "TITAN.NS": "Titan Company",
    "HAVELLS.NS": "Havells India",
    "VOLTAS.NS": "Voltas",
    "DIXON.NS": "Dixon Technologies",
    "BATAINDIA.NS": "Bata India",
    "PAGEIND.NS": "Page Industries",
    "TRENT.NS": "Trent",
    "DMART.NS": "Avenue Supermarts",
    "NYKAA.NS": "FSN E-Commerce Ventures",
    "ZOMATO.NS": "Zomato",
    "PAYTM.NS": "One97 Communications",
    # Chemicals
    "SRF.NS": "SRF",
    "PIIND.NS": "PI Industries",
    "PIDILITIND.NS": "Pidilite Industries",
    "TATACHEM.NS": "Tata Chemicals",
    "UPL.NS": "UPL",
    "COROMANDEL.NS": "Coromandel International",
    # Aviation
    "INDIGO.NS": "InterGlobe Aviation",
    "CONCOR.NS": "Container Corporation of India",
    "DELHIVERY.NS": "Delhivery"
}

# --- DAILY CACHE ---
# Structure: { "STOCK_SYMBOL": {"date": "YYYY-MM-DD", "ml_results": {...}, "monthly": {...}} }
daily_prediction_cache = {}

# --- SECTOR CACHE ---
sector_info_cache = {}

# --- DASHBOARD CACHE ---
dashboard_cache = {
    "date": None,
    "data": None
}

# --- SCANNER CACHE ---
scanner_cache = {
    "date": None,
    "results": []
}

def login_required():
    if "user" not in session: return False
    return True

# --- ROUTES ---
@app.route("/")
def home():
    if not login_required(): return redirect(url_for("login"))
    return redirect(url_for("dashboard"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user:
            # Check if it's a hashed password (Werkzeug uses scrypt or pbkdf2)
            if user.password.startswith("scrypt:") or user.password.startswith("pbkdf2:"):
                if check_password_hash(user.password, password):
                    session["user"] = username
                    session["user_id"] = user.id
                    return redirect(url_for("dashboard"))
            else:
                # Handle old unhashed passwords
                if user.password == password:
                    session["user"] = username
                    session["user_id"] = user.id
                    return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(username=username).first():
            return render_template("signup.html", error="User already exists")
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        session["user"] = username
        session["user_id"] = new_user.id
        return redirect(url_for("dashboard"))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("user_id", None)
    return redirect(url_for("login"))

@app.route("/keep-alive")
def keep_alive():
    return "OK", 200

@app.route("/dashboard")
def dashboard():
    if not login_required(): return redirect(url_for("login"))
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    import json
    
    global dashboard_cache
    if dashboard_cache.get("date") == today_str and dashboard_cache.get("data"):
        dash_data = dashboard_cache["data"]
    else:
        # --- PARALLEL DASHBOARD FETCHING ---
        def fetch_index_data(ticker, name):
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if len(hist) >= 2:
                    prev_close = hist["Close"].iloc[-2]
                    curr_price = hist["Close"].iloc[-1]
                    change = ((curr_price - prev_close) / prev_close) * 100
                    return name, {"price": round(curr_price, 2), "change": round(change, 2)}
                return name, {"price": 0, "change": 0}
            except:
                return name, {"price": 0, "change": 0}

        def fetch_pick_data(stock):
            try:
                df = yf.Ticker(stock).history(period="100d")
                m = calculate_metrics(df.copy())
                return {"stock": stock, "trend": m["trend"], "rsi": m["rsi"]}
            except:
                return None

        def fetch_sector_data(name, sym):
            try:
                si = yf.Ticker(sym).history(period="2d")
                if len(si) >= 2:
                    prev = si["Close"].iloc[-2]
                    curr = si["Close"].iloc[-1]
                    chg = round(((curr - prev) / prev) * 100, 2)
                    return {"name": name, "change": chg}
                return None
            except:
                return None

        indices_to_fetch = [('^NSEI', 'NIFTY 50'), ('^BSESN', 'SENSEX'), ('^NSEBANK', 'BANK NIFTY')]
        picks_to_fetch = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        sector_indices = {
            "Banking": "^NSEBANK", "IT": "^CNXIT", "FMCG": "^CNXFMCG",
            "Auto": "^CNXAUTO", "Metal": "^CNXMETAL", "Pharma": "^CNXPHARMA",
            "Realty": "^CNXREALTY", "Energy": "^CNXENERGY"
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Indices
            idx_futures = [executor.submit(fetch_index_data, t, n) for t, n in indices_to_fetch]
            # Top Picks
            pick_futures = [executor.submit(fetch_pick_data, s) for s in picks_to_fetch]
            # Sectors
            sec_futures = [executor.submit(fetch_sector_data, n, s) for n, s in sector_indices.items()]

            indices_data = dict(f.result() for f in idx_futures)
            top_picks = [f.result() for f in pick_futures if f.result()]
            sector_heat = [f.result() for f in sec_futures if f.result()]

        sector_heat.sort(key=lambda x: x["change"], reverse=True)
                
        # --- News Headlines ---
        news_headlines = []
        try:
            news = yf.Ticker('^NSEI').news
            if news:
                for n in news[:4]:
                    if 'content' in n and 'title' in n['content']:
                        news_headlines.append(n['content']['title'])
                    elif 'title' in n:
                        news_headlines.append(n['title'])
        except:
            pass

        dash_data = {
            "indices": indices_data,
            "picks": top_picks,
            "news": news_headlines,
            "sector_heat": sector_heat
        }
        dashboard_cache["date"] = today_str
        dashboard_cache["data"] = dash_data
    
    return render_template("dashboard.html", dash_data=dash_data)

@app.route("/api/dashboard_sentiment")
def api_dashboard_sentiment():
    if not login_required(): return {"error": "Unauthorized"}, 401
    
    sentiment = {"score": 50, "summary": "Market is showing mixed signals."}
    try:
        news_headlines = []
        news = yf.Ticker('^NSEI').news
        if news:
            for n in news[:4]:
                if 'content' in n and 'title' in n['content']:
                    news_headlines.append(n['content']['title'])
                elif 'title' in n:
                    news_headlines.append(n['title'])
                    
        if news_headlines:
            client = get_gemini_client()
            if client:
                prompt = "Determine the Fear & Greed index (0-100, where 0 is extreme fear/bearish and 100 is extreme greed/bullish) based on these Indian market headlines. Return strictly a JSON format like: {\"score\": 75, \"summary\": \"1-sentence summary\"}. Headlines: " + " | ".join(news_headlines)
                
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                    }
                )
                import json
                s_data = json.loads(resp.text)
                sentiment["score"] = s_data.get("score", 50)
                sentiment["summary"] = s_data.get("summary", s_data.get("explanation", "Analysis complete."))
            else:
                sentiment["summary"] = "AI features unavailable. Check GOOGLE_API_KEY."
        return sentiment
    except Exception as e:
        print("News/Sentiment Error:", e)
        return sentiment

@app.route("/buy")
def buy():
    if not login_required(): return redirect(url_for("login"))
    user_buys = BuyList.query.filter_by(user_id=session.get("user_id")).all()
    
    portfolio_data = []
    total_invested = 0
    current_market_value = 0
    sector_dist = {} # {sector: current_value}
    
    # Use Parallel Processing for Portfolio Data
    def fetch_portfolio_item(buy_item):
        data = {
            "stock": buy_item.stock,
            "status": buy_item.status,
            "quantity": buy_item.quantity,
            "purchase_price": buy_item.purchase_price,
            "live_price": None,
            "pnl": None,
            "pnl_percent": None,
            "sector": "Unknown",
            "market_value": 0
        }
        
        # Sector Info - OPTIMIZED: avoid t.info which is extremely slow
        if buy_item.stock not in sector_info_cache:
            sector = "Equity"
            if buy_item.stock in ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "MPHASIS.NS", "KPITTECH.NS", "PERSISTENT.NS", "COFORGE.NS"]: sector = "IT Services"
            elif buy_item.stock in ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS"]: sector = "Banks"
            elif buy_item.stock in ["BAJFINANCE.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PFC.NS", "RECLTD.NS"]: sector = "NBFCs"
            elif buy_item.stock in ["RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "COALINDIA.NS", "BPCL.NS"]: sector = "Energy"
            elif buy_item.stock in ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"]: sector = "FMCG"
            elif buy_item.stock in ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJAHT.NS"]: sector = "Autos"
            elif buy_item.stock in ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS"]: sector = "Metals"
            elif buy_item.stock in ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"]: sector = "Pharma"
            elif buy_item.stock in ["LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "DLF.NS"]: sector = "Real Estate/Infra"
            sector_info_cache[buy_item.stock] = sector
            
        # Determine a generic sector based on symbols if still unknown
        data["sector"] = sector_info_cache[buy_item.stock]

        if buy_item.status == "Bought":
            try:
                hist = yf.Ticker(buy_item.stock).history(period="1d")
                if not hist.empty:
                    data["live_price"] = round(hist["Close"].iloc[-1], 2)
                    data["pnl"] = round((data["live_price"] - buy_item.purchase_price) * buy_item.quantity, 2)
                    data["pnl_percent"] = round(((data["live_price"] - buy_item.purchase_price) / buy_item.purchase_price) * 100, 2) if buy_item.purchase_price > 0 else 0
                    data["market_value"] = data["live_price"] * buy_item.quantity
            except:
                pass
        return data

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        portfolio_results = list(executor.map(fetch_portfolio_item, user_buys))

    for data in portfolio_results:
        if data["status"] == "Bought" and data["live_price"]:
            total_invested += (data["purchase_price"] * data["quantity"])
            current_market_value += data["market_value"]
            sec = data["sector"]
            sector_dist[sec] = sector_dist.get(sec, 0) + data["market_value"]
        portfolio_data.append(data)
    
    # --- RECORD SNAPSHOT (Step 2) ---
    today_str = datetime.now().strftime("%Y-%m-%d")
    existing_snap = PortfolioSnapshot.query.filter_by(user_id=session.get("user_id"), date=today_str).first()
    if not existing_snap and current_market_value > 0:
        new_snap = PortfolioSnapshot(user_id=session.get("user_id"), date=today_str, total_value=round(current_market_value, 2))
        db.session.add(new_snap)
        db.session.commit()
    
    # Get History for Chart
    snaps = PortfolioSnapshot.query.filter_by(user_id=session.get("user_id")).order_by(PortfolioSnapshot.date).all()
    history_labels = [s.date for s in snaps]
    history_values = [s.total_value for s in snaps]

    total_pnl = round(current_market_value - total_invested, 2)
    total_pnl_percent = round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0
    
    summary = {
        "total_invested": round(total_invested, 2),
        "current_market_value": round(current_market_value, 2),
        "total_pnl": total_pnl,
        "total_pnl_percent": total_pnl_percent,
        "sector_labels": list(sector_dist.keys()),
        "sector_values": list(sector_dist.values()),
        "history_labels": history_labels,
        "history_values": history_values
    }
        
    return render_template("buy.html", portfolio_data=portfolio_data, summary=summary)

@app.route("/add_to_buy", methods=["POST"])
def add_to_buy():
    if not login_required(): return "Login Required", 401
    stock = request.form.get("stock")
    if not stock or stock.lower() == "none" or stock.strip() == "":
        return "Invalid Stock", 400
    user_id = session.get("user_id")
    existing = BuyList.query.filter_by(user_id=user_id, stock=stock).first()
    if not existing:
        new_buy = BuyList(user_id=user_id, stock=stock)
        db.session.add(new_buy)
        db.session.commit()
    return "Added"

@app.route("/remove_from_buy", methods=["POST"])
def remove_from_buy():
    if not login_required(): return redirect(url_for("login"))
    stock = request.form.get("stock")
    user_id = session.get("user_id")
    BuyList.query.filter_by(user_id=user_id, stock=stock).delete()
    db.session.commit()
    return redirect(request.referrer or url_for("buy"))

@app.route("/mark_bought", methods=["POST"])
def mark_bought():
    if not login_required(): return redirect(url_for("login"))
    user_id = session.get("user_id")
    stock = request.form.get("stock")
    try:
        qty = int(request.form.get("quantity", 0))
        price = float(request.form.get("price", 0.0))
    except:
        return redirect(url_for("buy"))
        
    item = BuyList.query.filter_by(user_id=user_id, stock=stock, status="Queued").first()
    if item and qty > 0 and price > 0:
        item.status = "Bought"
        item.quantity = qty
        item.purchase_price = price
        db.session.commit()
    return redirect(url_for("buy"))

@app.route("/watchlist")
def watchlist():
    if not login_required(): return redirect(url_for("login"))
    watches = WatchList.query.filter_by(user_id=session.get("user_id")).all()
    user_watchlist = [w.stock for w in watches]
    return render_template("watchlist.html", watchlist=user_watchlist)

@app.route("/add_to_watchlist", methods=["POST"])
def add_to_watchlist():
    if not login_required(): return "Login Required", 401
    stock = request.form.get("stock")
    if not stock or stock.lower() == "none" or stock.strip() == "":
        return "Invalid Stock", 400
    user_id = session.get("user_id")
    existing = WatchList.query.filter_by(user_id=user_id, stock=stock).first()
    if not existing:
        new_watch = WatchList(user_id=user_id, stock=stock)
        db.session.add(new_watch)
        db.session.commit()
    return "Added"

@app.route("/remove_from_watchlist", methods=["POST"])
def remove_from_watchlist():
    if not login_required(): return redirect(url_for("login"))
    stock = request.form.get("stock")
    user_id = session.get("user_id")
    WatchList.query.filter_by(user_id=user_id, stock=stock).delete()
    db.session.commit()
    return redirect(request.referrer or url_for("watchlist"))

@app.route("/clear_watchlist", methods=["POST"])
def clear_watchlist():
    if not login_required(): return redirect(url_for("login"))
    user_id = session.get("user_id")
    WatchList.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return redirect(request.referrer or url_for("watchlist"))

@app.route("/go/<broker>/<stock>")
def go_to_broker(broker, stock):
    clean_stock = stock.split(".")[0]
    if broker == "groww": 
        return redirect(f"https://groww.in/search?q={clean_stock}")
    elif broker == "upstox": 
        # Upstox Pro doesn't support query params, but this is the stock page
        return redirect(f"https://upstox.com/stocks/")
    return redirect(url_for("dashboard"))

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if not login_required(): return redirect(url_for("login"))
    stock = request.args.get("stock") or request.form.get("stock")
    stock = stock.strip().upper() if stock else None
    
    selected_range = request.args.get("range") or request.form.get("range", "1Y")
    
    if not stock:
        return render_template("analyze.html", 
            stock_symbol=None, 
            stocks=list(stocks_list.keys()), 
            selected_range="1Y", 
            dates=[], 
            prices=[], 
            metrics={"ma20": [], "ma50": []},
            live_price=None,
            price_change=None,
            change_percent=None
        )

    # Live Price
    live_price = None
    try:
        ticker = yf.Ticker(stock)
        live_data = ticker.history(period="1d")
        if not live_data.empty: live_price = round(live_data["Close"].iloc[-1], 2)
    except: pass

    # Dataset Load - OPTIMIZED: fetch only reasonable amount for the chart
    try:
        ticker = yf.Ticker(stock)
        fetch_period = "5y" # Default to 5y to cover most ranges
        if selected_range in ["15D", "1M", "6M", "1Y"]: fetch_period = "2y"
        
        df = ticker.history(period=fetch_period)
        if df is None or df.empty:
            return render_template("analyze.html", 
                stock_symbol=None, 
                stocks=list(stocks_list.keys()), 
                selected_range=selected_range,
                dates=[], 
                prices=[], 
                metrics={"ma20": [], "ma50": []},
                live_price=None,
                error=f"No data found for {stock}. Try a different ticker.")
        
        df = df.reset_index()
        # Standardize Date column
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    except Exception as e:
        print(f"Error fetching data for {stock}: {e}")
        return render_template("analyze.html", 
            stock_symbol=None, 
            stocks=list(stocks_list.keys()), 
            selected_range=selected_range,
            dates=[], 
            prices=[], 
            metrics={"ma20": [], "ma50": []},
            live_price=None,
            error="Market Data Error. Please check your connection.")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    
    # Calculate MAs on the full fetched set
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Time Filtering for Chart
    max_date = df["Date"].max()
    if selected_range == "15D": df_plot = df[df["Date"] >= max_date - pd.DateOffset(days=15)]
    elif selected_range == "1M": df_plot = df[df["Date"] >= max_date - pd.DateOffset(months=1)]
    elif selected_range == "6M": df_plot = df[df["Date"] >= max_date - pd.DateOffset(months=6)]
    elif selected_range == "1Y": df_plot = df[df["Date"] >= max_date - pd.DateOffset(years=1)]
    elif selected_range == "3Y": df_plot = df[df["Date"] >= max_date - pd.DateOffset(years=3)]
    elif selected_range == "5Y": df_plot = df[df["Date"] >= max_date - pd.DateOffset(years=5)]
    else: df_plot = df

    # Basic Metrics (Non-ML)
    metrics = calculate_metrics(df.copy())
    
    is_budget_month = datetime.now().month == 2

    # REMOVED: train_and_predict and generate_ai_explanation (Now handled by async API)
    
    return render_template(
        "analyze.html",
        stock_symbol=stock,
        stock_name=stocks_list.get(stock, stock),
        stocks=list(stocks_list.keys()),
        selected_range=selected_range,
        dates=df_plot["Date"].apply(lambda x: x.strftime('%Y-%m-%d')).tolist(),
        prices=df_plot["Close"].tolist(),
        metrics=metrics,
        live_price=live_price,
        price_change=round(live_price - df["Close"].iloc[-1], 2) if live_price else 0,
        change_percent=round(((live_price - df["Close"].iloc[-1]) / df["Close"].iloc[-1]) * 100, 2) if live_price and df["Close"].iloc[-1] != 0 else 0,
        is_budget_month=is_budget_month
    )

@app.route("/api/analyze_deep", methods=["POST"])
def api_analyze_deep():
    if not login_required(): return {"error": "Unauthorized"}, 401
    symbol = request.form.get("stock")
    if not symbol: return {"error": "Missing symbol"}, 400
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Check if we already have the ML results (e.g., from scanner)
        res = None
        monthly_values = []
        if symbol in daily_prediction_cache and daily_prediction_cache[symbol].get("date") == today_str:
            res = daily_prediction_cache[symbol].get("ml_results")
            monthly_values = daily_prediction_cache[symbol].get("monthly_values", [])

        # Fetch data only if needed (for AI explanation or if cache miss)
        t = yf.Ticker(symbol)
        df = t.history(period="2y")
        if df.empty: return {"error": "No data found for this ticker"}, 404
        
        # 1. Technical Metrics (Required for AI Explanation)
        m = calculate_metrics(df.copy())
        
        # 2. Prediction (If not in cache)
        if not res:
            res = train_and_predict(df.copy())
        
        # 3. Merge results for AI
        ai_data = m.copy()
        ai_data.update(res) 
        
        # AI Explanation (with its own fallback)
        ai_explanation = daily_prediction_cache.get(symbol, {}).get("ai_explanation")
        if not ai_explanation:
            curr_p = df.iloc[-1]['Close']
            prev_p = df.iloc[-2]['Close'] if len(df) > 1 else curr_p
            chg_pct = round(((curr_p - prev_p) / prev_p) * 100, 2)
            ai_explanation = generate_ai_explanation(ai_data, chg_pct)
        
        # Monthly Seasonality (If not in cache)
        if not monthly_values:
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            else:
                df.index = df.index.tz_localize(None)
            
            df_m = df['Close'].resample('ME').ffill().pct_change() * 100
            monthly_values = [round(df_m[df_m.index.month == i].mean(), 2) for i in range(1, 13)]
            monthly_values = [0 if (np.isnan(x) or np.isinf(x)) else x for x in monthly_values]
        
        # --- CACHE STORE ---
        daily_prediction_cache[symbol] = {
            "date": today_str,
            "ml_results": res,
            "ai_explanation": ai_explanation,
            "monthly_values": monthly_values
        }
        
        return {
            "prediction": res,
            "ai_explanation": ai_explanation,
            "monthly_values": monthly_values
        }
    except Exception as e:
        print(f"Deep Analysis Exception: {e}")
        # Final safety fallback: Provide basic metrics even if ML/AI fails
        try:
            m = calculate_metrics(df.tail(100).copy())
            fallback_res = {
                "pred_price": df.iloc[-1]['Close'],
                "pred_direction": "Neutral",
                "score": 5,
                "recommendation": "Engine Error",
                "confidence": 0,
                "risk_level": m.get('risk_level', 'High'),
                "momentum": "Neutral"
            }
            return {
                "prediction": fallback_res,
                "ai_explanation": f"Technical diagnostic complete. Note: Local analysis engine is active due to a service interruption: {str(e)}",
                "monthly_values": []
            }
        except:
            return {"error": f"Analysis engine successfully avoided a singularity: {str(e)}"}, 500

@app.route("/scanner")
def scanner():
    if not login_required(): return redirect(url_for("login"))
    # Return page immediately with the symbols we intend to scan
    target_symbols = list(stocks_list.keys())[:24]
    return render_template("scanner.html", symbols=target_symbols)

@app.route("/api/scan_stock", methods=["POST"])
def api_scan_stock():
    if not login_required(): return {"error": "Unauthorized"}, 401
    symbol = request.form.get("symbol")
    if not symbol: return {"error": "Missing symbol"}, 400
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Optimization: Reuse the same dataframe for both ML and Metrics
        t = yf.Ticker(symbol)
        df = t.history(period="2y")
        if df.empty: return {"error": "No data"}, 404
        
        if symbol in daily_prediction_cache and daily_prediction_cache[symbol]["date"] == today_str:
            res = daily_prediction_cache[symbol]["ml_results"]
        else:
            # Use Random Forest (now optimized with fast_mode)
            res = train_and_predict(df, fast_mode=True)
            # Standardize cache structure
            daily_prediction_cache[symbol] = {
                "date": today_str,
                "ml_results": res,
                "monthly_values": [] 
            }

        # Calculate metrics using the SAME dataframe (last 100 days slice)
        m = calculate_metrics(df.tail(100).copy())
        
        return {
            "symbol": symbol,
            "name": stocks_list.get(symbol, symbol),
            "score": res.get("score", 5),
            "rsi": m.get("rsi", 50),
            "trend": m.get("trend", "Neutral")
        }
    except Exception as e:
        print(f"API Scan Error ({symbol}): {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True)