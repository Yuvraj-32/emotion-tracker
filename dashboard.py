"""
==========================================================
  EMOTION TRACKER — WEB DASHBOARD
==========================================================
  Two pages:
    /           → Live session dashboard (current session)
    /analytics  → Multi-day analytics (all sessions)

  Usage:
    python dashboard.py
    Then open http://localhost:5000 in your browser.
==========================================================
"""

import os
import csv
import glob
import json
from datetime import datetime
from collections import Counter, defaultdict
from flask import Flask, render_template_string

app = Flask(__name__)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ══════════════════════════════════════════════
#  SHARED STYLES + NAV
# ══════════════════════════════════════════════
SHARED_HEAD = """
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
        min-height: 100vh;
    }
    .header {
        text-align: center;
        padding: 25px 20px 12px;
        background: rgba(0,0,0,0.3);
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .header h1 {
        font-size: 26px; font-weight: 700;
        background: linear-gradient(to right, #00e676, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header p { color: #999; font-size: 13px; margin-top: 4px; }
    .nav {
        display: flex; justify-content: center; gap: 12px;
        padding: 12px; background: rgba(0,0,0,0.2);
    }
    .nav a {
        color: #aaa; text-decoration: none; padding: 8px 20px;
        border-radius: 8px; font-size: 14px; font-weight: 600;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.2s;
    }
    .nav a:hover { background: rgba(255,255,255,0.1); color: #fff; }
    .nav a.active {
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border-color: #00e676;
    }
    .stats-row {
        display: flex; justify-content: center; gap: 18px;
        padding: 22px 20px; flex-wrap: wrap;
    }
    .stat-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px; padding: 18px 26px;
        text-align: center; min-width: 145px;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-card .value { font-size: 32px; font-weight: 700; }
    .stat-card .label {
        font-size: 11px; text-transform: uppercase;
        letter-spacing: 1px; color: #888; margin-top: 4px;
    }
    .charts-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
        padding: 0 25px 25px; max-width: 1200px; margin: 0 auto;
    }
    .chart-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px; padding: 18px;
    }
    .chart-box h3 {
        font-size: 15px; font-weight: 600;
        margin-bottom: 12px; color: #ccc;
    }
    .chart-box.full { grid-column: 1 / -1; }
    .no-data {
        text-align: center; padding: 80px 20px;
        color: #666; font-size: 18px;
    }
    .green  { color: #00e676; }
    .red    { color: #ff5252; }
    .orange { color: #ffab40; }
    .blue   { color: #40c4ff; }
    .purple { color: #bb86fc; }
    @media (max-width: 768px) { .charts-grid { grid-template-columns: 1fr; } }
</style>
"""

NAV_LIVE = """
<div class="nav">
    <a href="/" class="active">📊 Live Session</a>
    <a href="/analytics">📈 Multi-Day Analytics</a>
</div>
"""

NAV_ANALYTICS = """
<div class="nav">
    <a href="/">📊 Live Session</a>
    <a href="/analytics" class="active">📈 Multi-Day Analytics</a>
</div>
"""

# ══════════════════════════════════════════════
#  LIVE SESSION TEMPLATE
# ══════════════════════════════════════════════
LIVE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    """ + SHARED_HEAD + """
    <meta http-equiv="refresh" content="5">
    <title>Emotion Tracker — Live Session</title>
</head>
<body>
    <div class="header">
        <h1>🧠 Emotion & Attention Dashboard</h1>
        <p>Session: {{ session_file }} &nbsp;|&nbsp; Auto-refreshes every 5 seconds</p>
    </div>
    """ + NAV_LIVE + """

    {% if has_data %}
    <div class="stats-row">
        <div class="stat-card">
            <div class="value green">{{ stats.data_points }}</div>
            <div class="label">Data Points</div>
        </div>
        <div class="stat-card">
            <div class="value blue">{{ stats.avg_engagement }}</div>
            <div class="label">Avg Engagement</div>
        </div>
        <div class="stat-card">
            <div class="value red">{{ stats.distractions }}</div>
            <div class="label">Distractions</div>
        </div>
        <div class="stat-card">
            <div class="value orange">{{ stats.drowsy_events }}</div>
            <div class="label">Drowsy Events</div>
        </div>
        <div class="stat-card">
            <div class="value">{{ stats.top_emotion }}</div>
            <div class="label">Top Emotion</div>
        </div>
        {% if stats.productivity != 'N/A' %}
        <div class="stat-card">
            <div class="value {% if stats.productivity == 'HIGH' %}green{% elif stats.productivity == 'LOW' %}red{% else %}orange{% endif %}">
                {{ stats.productivity }}
            </div>
            <div class="label">ML Productivity</div>
        </div>
        {% endif %}
    </div>

    <div class="charts-grid">
        <div class="chart-box">
            <h3>Emotion Distribution</h3>
            <canvas id="emotionPie"></canvas>
        </div>
        <div class="chart-box">
            <h3>Attention vs Distraction</h3>
            <canvas id="attentionPie"></canvas>
        </div>
        <div class="chart-box full">
            <h3>Engagement Over Time</h3>
            <canvas id="engagementLine"></canvas>
        </div>
    </div>

    <script>
        const COLORS = {
            'Happy': '#00e676', 'Neutral': '#40c4ff',
            'Tired / Sad': '#ffab40', 'Frustrated': '#ff5252',
            'Surprised': '#ffff00', 'Scanning...': '#999'
        };
        new Chart(document.getElementById('emotionPie'), {
            type: 'doughnut',
            data: {
                labels: {{ emo_labels | safe }},
                datasets: [{
                    data: {{ emo_values | safe }},
                    backgroundColor: {{ emo_labels | safe }}.map(l => COLORS[l] || '#888'),
                    borderWidth: 0
                }]
            },
            options: { plugins: { legend: { labels: { color: '#ccc' } } }, cutout: '60%' }
        });
        new Chart(document.getElementById('attentionPie'), {
            type: 'doughnut',
            data: {
                labels: ['Focused', 'Distracted', 'Drowsy'],
                datasets: [{
                    data: [{{ stats.focused }}, {{ stats.distractions }}, {{ stats.drowsy_events }}],
                    backgroundColor: ['#00e676', '#ff5252', '#ffab40'],
                    borderWidth: 0
                }]
            },
            options: { plugins: { legend: { labels: { color: '#ccc' } } }, cutout: '60%' }
        });
        new Chart(document.getElementById('engagementLine'), {
            type: 'line',
            data: {
                labels: {{ eng_times | safe }},
                datasets: [{
                    label: 'Engagement Score',
                    data: {{ eng_values | safe }},
                    borderColor: '#00e676',
                    backgroundColor: 'rgba(0, 230, 118, 0.1)',
                    borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3
                }]
            },
            options: {
                scales: {
                    y: { min: 0, max: 100, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#888', maxTicksLimit: 20 }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { labels: { color: '#ccc' } } }
            }
        });
    </script>

    {% else %}
    <div class="no-data">
        <p>No session data found yet.</p>
        <p style="margin-top:10px; font-size:14px; color:#555;">
            Run <code style="color:#00e676;">python main.py</code> first, then refresh.
        </p>
    </div>
    {% endif %}
</body>
</html>
"""

# ══════════════════════════════════════════════
#  MULTI-DAY ANALYTICS TEMPLATE
# ══════════════════════════════════════════════
ANALYTICS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    """ + SHARED_HEAD + """
    <meta http-equiv="refresh" content="30">
    <title>Emotion Tracker — Multi-Day Analytics</title>
</head>
<body>
    <div class="header">
        <h1>📈 Multi-Day Productivity Analytics</h1>
        <p>Aggregated across {{ stats.total_sessions }} session(s) &nbsp;|&nbsp; {{ stats.total_days }} day(s) tracked</p>
    </div>
    """ + NAV_ANALYTICS + """

    {% if has_data %}
    <div class="stats-row">
        <div class="stat-card">
            <div class="value green">{{ stats.total_sessions }}</div>
            <div class="label">Sessions</div>
        </div>
        <div class="stat-card">
            <div class="value blue">{{ stats.total_hours }}</div>
            <div class="label">Hours Tracked</div>
        </div>
        <div class="stat-card">
            <div class="value green">{{ stats.avg_engagement }}</div>
            <div class="label">Avg Engagement</div>
        </div>
        <div class="stat-card">
            <div class="value">{{ stats.focus_rate }}%</div>
            <div class="label">Focus Rate</div>
        </div>
        <div class="stat-card">
            <div class="value green">{{ stats.best_day }}</div>
            <div class="label">Best Day</div>
        </div>
        <div class="stat-card">
            <div class="value red">{{ stats.worst_day }}</div>
            <div class="label">Worst Day</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-box full">
            <h3>Daily Engagement Trend</h3>
            <canvas id="dailyEngagement"></canvas>
        </div>
        <div class="chart-box">
            <h3>Daily Focus Rate</h3>
            <canvas id="dailyFocus"></canvas>
        </div>
        <div class="chart-box">
            <h3>Productivity Distribution</h3>
            <canvas id="productivityPie"></canvas>
        </div>
        <div class="chart-box">
            <h3>Emotion Trends (All Sessions)</h3>
            <canvas id="emotionTrend"></canvas>
        </div>
        <div class="chart-box">
            <h3>Peak Productivity Hours</h3>
            <canvas id="hourlyChart"></canvas>
        </div>
    </div>

    {% if model_info %}
    <div class="stats-row">
        <div class="stat-card" style="min-width:400px; text-align:left;">
            <div class="label" style="margin-bottom:10px;">🤖 ML Model Info</div>
            <div style="font-size:13px; line-height:1.8;">
                Accuracy: <span class="green">{{ model_info.accuracy }}%</span><br>
                Trained on: {{ model_info.num_samples }} samples<br>
                Top feature: <span class="blue">{{ model_info.top_feature }}</span><br>
                Last trained: {{ model_info.trained_at }}
            </div>
        </div>
    </div>
    {% endif %}

    <script>
        // Daily engagement line chart
        new Chart(document.getElementById('dailyEngagement'), {
            type: 'line',
            data: {
                labels: {{ daily_dates | safe }},
                datasets: [{
                    label: 'Avg Engagement',
                    data: {{ daily_engagement | safe }},
                    borderColor: '#00e676',
                    backgroundColor: 'rgba(0, 230, 118, 0.1)',
                    borderWidth: 2, pointRadius: 4, fill: true, tension: 0.3
                }]
            },
            options: {
                scales: {
                    y: { min: 0, max: 100, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { labels: { color: '#ccc' } } }
            }
        });

        // Daily focus rate bar chart
        new Chart(document.getElementById('dailyFocus'), {
            type: 'bar',
            data: {
                labels: {{ daily_dates | safe }},
                datasets: [{
                    label: 'Focus Rate %',
                    data: {{ daily_focus | safe }},
                    backgroundColor: {{ daily_focus | safe }}.map(v => v >= 70 ? '#00e676' : v >= 40 ? '#ffab40' : '#ff5252'),
                    borderRadius: 6
                }]
            },
            options: {
                scales: {
                    y: { min: 0, max: 100, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { display: false } }
            }
        });

        // Productivity distribution pie
        new Chart(document.getElementById('productivityPie'), {
            type: 'doughnut',
            data: {
                labels: ['High', 'Medium', 'Low'],
                datasets: [{
                    data: [{{ prod_high }}, {{ prod_med }}, {{ prod_low }}],
                    backgroundColor: ['#00e676', '#ffab40', '#ff5252'],
                    borderWidth: 0
                }]
            },
            options: { plugins: { legend: { labels: { color: '#ccc' } } }, cutout: '60%' }
        });

        // Emotion trends bar chart
        new Chart(document.getElementById('emotionTrend'), {
            type: 'bar',
            data: {
                labels: {{ emo_labels | safe }},
                datasets: [{
                    label: 'Count',
                    data: {{ emo_values | safe }},
                    backgroundColor: {{ emo_labels | safe }}.map(l => ({
                        'Happy': '#00e676', 'Neutral': '#40c4ff',
                        'Tired / Sad': '#ffab40', 'Frustrated': '#ff5252',
                        'Surprised': '#ffff00', 'Scanning...': '#999'
                    }[l]) || '#888'),
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { display: false } }
            }
        });

        // Peak hours bar chart
        new Chart(document.getElementById('hourlyChart'), {
            type: 'bar',
            data: {
                labels: {{ hourly_labels | safe }},
                datasets: [{
                    label: 'Avg Engagement',
                    data: {{ hourly_values | safe }},
                    backgroundColor: {{ hourly_values | safe }}.map(v => v >= 65 ? '#00e676' : v >= 35 ? '#ffab40' : '#ff5252'),
                    borderRadius: 6
                }]
            },
            options: {
                scales: {
                    y: { min: 0, max: 100, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    </script>

    {% else %}
    <div class="no-data">
        <p>Not enough data for multi-day analytics.</p>
        <p style="margin-top:10px; font-size:14px; color:#555;">
            Run a few more tracking sessions to see trends.
        </p>
    </div>
    {% endif %}
</body>
</html>
"""


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def get_latest_csv():
    pattern = os.path.join(LOG_DIR, "session_*.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def get_all_csvs():
    pattern = os.path.join(LOG_DIR, "session_*.csv")
    return sorted(glob.glob(pattern))


def parse_csv(path):
    rows = []
    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    return rows


def extract_date_from_filename(path):
    """Extract YYYY-MM-DD from session_YYYY-MM-DD_HH-MM-SS.csv"""
    fname = os.path.basename(path)
    try:
        return fname.replace("session_", "")[:10]
    except Exception:
        return "unknown"


def extract_hour_from_timestamp(ts):
    """Extract hour from HH:MM:SS.mmm timestamp"""
    try:
        return int(ts.split(":")[0])
    except Exception:
        return 0


# ══════════════════════════════════════════════
#  LIVE SESSION ROUTE
# ══════════════════════════════════════════════
@app.route('/')
def live_dashboard():
    csv_path = get_latest_csv()

    if not csv_path:
        return render_template_string(LIVE_TEMPLATE,
                                      has_data=False, session_file="None")

    rows = parse_csv(csv_path)
    if not rows:
        return render_template_string(LIVE_TEMPLATE,
                                      has_data=False,
                                      session_file=os.path.basename(csv_path))

    # Stats
    emotions = Counter(r['emotion'] for r in rows)
    distractions = sum(1 for r in rows if r.get('is_looking') == 'False')
    drowsy = sum(1 for r in rows if r.get('is_drowsy') == 'True')
    focused = len(rows) - distractions - drowsy

    engagements = []
    for r in rows:
        try:
            engagements.append(float(r['engagement']))
        except (ValueError, KeyError):
            pass
    avg_eng = int(sum(engagements) / len(engagements)) if engagements else 0
    top_emo = emotions.most_common(1)[0][0] if emotions else "N/A"

    # Latest productivity prediction
    latest_prod = 'N/A'
    for r in reversed(rows):
        p = r.get('productivity', 'N/A')
        if p and p != 'N/A':
            latest_prod = p
            break

    # Timeline
    step = max(1, len(rows) // 500)
    eng_times = [rows[i]['timestamp'] for i in range(0, len(rows), step)]
    eng_values = []
    for i in range(0, len(rows), step):
        try:
            eng_values.append(float(rows[i]['engagement']))
        except (ValueError, KeyError):
            eng_values.append(0)

    emo_labels = list(emotions.keys())
    emo_values = list(emotions.values())

    stats = {
        'data_points': len(rows),
        'avg_engagement': avg_eng,
        'distractions': distractions,
        'drowsy_events': drowsy,
        'focused': max(0, focused),
        'top_emotion': top_emo,
        'productivity': latest_prod,
    }

    return render_template_string(
        LIVE_TEMPLATE,
        has_data=True,
        session_file=os.path.basename(csv_path),
        stats=stats,
        emo_labels=emo_labels,
        emo_values=emo_values,
        eng_times=eng_times,
        eng_values=eng_values,
    )


# ══════════════════════════════════════════════
#  MULTI-DAY ANALYTICS ROUTE
# ══════════════════════════════════════════════
@app.route('/analytics')
def analytics_dashboard():
    files = get_all_csvs()

    if not files:
        return render_template_string(ANALYTICS_TEMPLATE,
                                      has_data=False,
                                      stats={'total_sessions': 0, 'total_days': 0})

    # Aggregate all data
    all_rows = []
    daily_data = defaultdict(list)  # date -> [rows]

    for fpath in files:
        date = extract_date_from_filename(fpath)
        rows = parse_csv(fpath)
        for row in rows:
            row['_date'] = date
        all_rows.extend(rows)
        daily_data[date].extend(rows)

    if not all_rows:
        return render_template_string(ANALYTICS_TEMPLATE,
                                      has_data=False,
                                      stats={'total_sessions': 0, 'total_days': 0})

    # ── Per-day aggregation ──
    sorted_dates = sorted(daily_data.keys())
    daily_engagement = []
    daily_focus = []
    best_day_score, worst_day_score = -1, 101
    best_day, worst_day = 'N/A', 'N/A'

    for date in sorted_dates:
        day_rows = daily_data[date]
        engs = []
        focused_count = 0
        total_count = len(day_rows)

        for r in day_rows:
            try:
                engs.append(float(r.get('engagement', 0)))
            except ValueError:
                pass
            if r.get('is_looking') == 'True' and r.get('is_drowsy') != 'True':
                focused_count += 1

        avg = int(sum(engs) / len(engs)) if engs else 0
        focus_pct = round(focused_count / total_count * 100) if total_count else 0
        daily_engagement.append(avg)
        daily_focus.append(focus_pct)

        if avg > best_day_score:
            best_day_score = avg
            best_day = date
        if avg < worst_day_score:
            worst_day_score = avg
            worst_day = date

    # ── Global stats ──
    all_engagements = []
    total_focused = 0
    for r in all_rows:
        try:
            all_engagements.append(float(r.get('engagement', 0)))
        except ValueError:
            pass
        if r.get('is_looking') == 'True' and r.get('is_drowsy') != 'True':
            total_focused += 1

    avg_eng = int(sum(all_engagements) / len(all_engagements)) if all_engagements else 0
    focus_rate = round(total_focused / len(all_rows) * 100) if all_rows else 0

    # Total hours = total data points × 5 seconds each
    total_hours = round(len(all_rows) * 5 / 3600, 1)

    # ── Productivity distribution ──
    prod_high = sum(1 for r in all_rows if r.get('productivity') == 'HIGH')
    prod_med = sum(1 for r in all_rows if r.get('productivity') == 'MEDIUM')
    prod_low = sum(1 for r in all_rows if r.get('productivity') == 'LOW')

    # If no ML predictions yet, derive from engagement
    if prod_high + prod_med + prod_low == 0:
        for r in all_rows:
            try:
                eng = float(r.get('engagement', 50))
                if eng >= 65:
                    prod_high += 1
                elif eng >= 35:
                    prod_med += 1
                else:
                    prod_low += 1
            except ValueError:
                prod_med += 1

    # ── Emotion distribution (all time) ──
    emotions = Counter(r['emotion'] for r in all_rows if r.get('emotion'))
    emo_labels = [e for e, _ in emotions.most_common(6)]
    emo_values = [c for _, c in emotions.most_common(6)]

    # ── Hourly productivity ──
    hourly_eng = defaultdict(list)
    for r in all_rows:
        try:
            hour = extract_hour_from_timestamp(r.get('timestamp', '00:00:00'))
            hourly_eng[hour].append(float(r.get('engagement', 0)))
        except ValueError:
            pass

    hourly_labels = []
    hourly_values = []
    for h in range(24):
        if h in hourly_eng:
            hourly_labels.append(f"{h:02d}:00")
            hourly_values.append(int(sum(hourly_eng[h]) / len(hourly_eng[h])))

    # ── ML Model metadata ──
    model_info = None
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            fi = meta.get('feature_importance', {})
            top_feature = list(fi.keys())[0] if fi else 'N/A'
            model_info = {
                'accuracy': round(meta.get('accuracy', 0) * 100, 1),
                'num_samples': meta.get('num_samples', 0),
                'top_feature': top_feature,
                'trained_at': meta.get('trained_at', 'N/A')[:16],
            }
        except Exception:
            pass

    stats = {
        'total_sessions': len(files),
        'total_days': len(sorted_dates),
        'total_hours': total_hours,
        'avg_engagement': avg_eng,
        'focus_rate': focus_rate,
        'best_day': best_day,
        'worst_day': worst_day,
    }

    return render_template_string(
        ANALYTICS_TEMPLATE,
        has_data=True,
        stats=stats,
        daily_dates=sorted_dates,
        daily_engagement=daily_engagement,
        daily_focus=daily_focus,
        prod_high=prod_high,
        prod_med=prod_med,
        prod_low=prod_low,
        emo_labels=emo_labels,
        emo_values=emo_values,
        hourly_labels=hourly_labels,
        hourly_values=hourly_values,
        model_info=model_info,
    )


if __name__ == '__main__':
    print("=" * 50)
    print("  EMOTION TRACKER — WEB DASHBOARD")
    print("=" * 50)
    print("  Live Session:  http://localhost:5000")
    print("  Analytics:     http://localhost:5000/analytics")
    print("  Auto-refreshes every 5 seconds")
    print("=" * 50)
    app.run(debug=False, port=5000)
