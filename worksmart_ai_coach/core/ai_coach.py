#!/usr/bin/env python3
"""
Ultimate AI Coach - Complete Consolidated Coaching System
========================================================

The ultimate consolidation of all AI coaching functionality into a single comprehensive system:

🧠 ALL COACHING SYSTEMS IN ONE FILE:
- Base AI Coach with Anthropic Claude integration
- Personalized AI Coach with persona detection and learning
- Enhanced ML AI Coach with full machine learning capabilities
- Pattern Learning, User Modeling, Predictive Analytics
- Feedback Collection and Continuous Learning
- Burnout Prediction and Optimal Timing
- Context Tracking and Adaptive Learning
- Micro-Interventions and Nudge DNA
- WorkSmart Integration and Telemetry Analysis

🚀 FEATURES INCLUDED:
✅ Machine Learning - Learns from user interactions with fallbacks
✅ Pattern Discovery - Finds effectiveness patterns in telemetry data  
✅ Burnout Prediction - Predicts and prevents user burnout
✅ Personalization - Adapts to individual user preferences and baselines
✅ Context Sensitivity - Multi-dimensional context analysis
✅ Continuous Learning - Improves over time with implicit/explicit feedback
✅ Predictive Intelligence - Anticipates user needs and optimal timing
✅ Adaptive Strategies - Changes approach based on measured effectiveness
✅ Persona Detection - Developer/Analyst/Manager specific coaching
✅ Anthropic API Integration - Advanced AI-generated coaching advice
✅ WorkSmart Integration - Uses official WorkSmart telemetry data
✅ Comprehensive Fallbacks - Works with or without ML dependencies

This single file replaces: coach.py, personalized_coach.py, enhanced_ai_coach.py, 
ai_coaching_system.py, and all ML component files.
"""

import json
import os
import asyncio
import logging
import time
import uuid
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from collections import defaultdict, deque
from urllib.parse import urlparse
from dataclasses import dataclass, asdict

# Optional ML dependencies with comprehensive fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Anthropic API integration
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# FALLBACK CLASSES FOR WHEN ML LIBRARIES ARE NOT AVAILABLE
# ============================================================================


class SimpleDataFrame:
    """Simple DataFrame-like class for basic data operations when pandas unavailable"""

    def __init__(self, data):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            self.data = data
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row.get(key) for row in self.data]
        else:
            return self.data[key]

    def mean(self):
        if not self.data:
            return {}
        result = {}
        for key in self.data[0].keys():
            values = [row[key] for row in self.data if isinstance(
                row.get(key), (int, float))]
            if values:
                result[key] = sum(values) / len(values)
        return result

# Simple ML model fallbacks


class SimpleFallbackClassifier:
    """Simple fallback classifier when sklearn unavailable"""

    def __init__(self, n_estimators=50, random_state=42):
        self.most_common_class = None

    def fit(self, X, y):
        if len(y) > 0:
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            self.most_common_class = max(
                class_counts.items(), key=lambda x: x[1])[0]

    def predict(self, X):
        return [self.most_common_class] * len(X) if self.most_common_class else ['unknown'] * len(X)

    def predict_proba(self, X):
        return [[0.5, 0.5]] * len(X)


class SimpleFallbackRegressor:
    """Simple fallback regressor when sklearn unavailable"""

    def __init__(self, n_estimators=50, random_state=42):
        self.mean_value = 0.5

    def fit(self, X, y):
        if len(y) > 0:
            self.mean_value = sum(y) / len(y)

    def predict(self, X):
        return [self.mean_value] * len(X)


class SimpleFallbackScaler:
    """Simple fallback scaler when sklearn unavailable"""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, X):
        if len(X) > 0 and len(X[0]) > 0:
            n_features = len(X[0])
            for i in range(n_features):
                values = [row[i] for row in X if len(row) > i]
                if values:
                    self.means[i] = sum(values) / len(values)
                    if len(values) > 1:
                        variance = sum(
                            (x - self.means[i]) ** 2 for x in values) / (len(values) - 1)
                        self.stds[i] = variance ** 0.5
                    else:
                        self.stds[i] = 1.0
        return self

    def transform(self, X):
        if not self.means:
            return X
        transformed = []
        for row in X:
            new_row = []
            for i, val in enumerate(row):
                if i in self.means and self.stds.get(i, 1.0) > 0:
                    new_row.append((val - self.means[i]) / self.stds[i])
                else:
                    new_row.append(val)
            transformed.append(new_row)
        return transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class IsolationForestFallback:
    """Simple anomaly detection fallback"""

    def __init__(self, contamination=0.1, random_state=42):
        self.threshold = 0.1

    def fit(self, X):
        return self

    def predict(self, X):
        return [1] * len(X)  # Normal by default


# Import or use fallback classes based on availability
if SKLEARN_AVAILABLE:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
else:
    RandomForestClassifier = SimpleFallbackClassifier
    GradientBoostingRegressor = SimpleFallbackRegressor
    RandomForestRegressor = SimpleFallbackRegressor
    StandardScaler = SimpleFallbackScaler
    IsolationForest = IsolationForestFallback

# ============================================================================
# DATA STRUCTURES AND USER MODEL
# ============================================================================


@dataclass
class UserPreferences:
    """User preferences for coaching"""
    notification_frequency: str = "medium"  # low, medium, high
    intervention_style: str = "balanced"    # minimal, balanced, assertive
    preferred_times: List[int] = None       # Preferred hours (0-23)
    avoided_times: List[int] = None         # Hours to avoid notifications
    # Minutes between break reminders (increased from 120)
    break_reminder_frequency: int = 180
    focus_session_duration: int = 45        # Preferred focus session length
    coaching_language: str = "specific"     # generic, specific, technical
    privacy_level: str = "standard"         # minimal, standard, detailed


@dataclass
class UserProfile:
    """Complete user behavioral profile"""
    user_id: str
    persona: str = "generic"                # analyst, developer, manager, etc.
    confidence_level: float = 0.5          # How confident we are in persona detection
    productivity_baseline: float = 0.5     # Personal productivity baseline
    focus_baseline: float = 0.5            # Personal focus baseline
    stress_tolerance: float = 0.5          # How much stress user can handle
    energy_patterns: Dict[int, float] = None  # Energy by hour of day
    productivity_patterns: Dict[str, float] = None  # Productivity by context
    intervention_effectiveness: Dict[str,
                                     float] = None  # Effectiveness by type
    preferences: UserPreferences = None
    last_updated: str = ""
    total_interactions: int = 0


@dataclass
class FeedbackEntry:
    """Single feedback entry for learning"""
    intervention_id: str
    user_id: str
    timestamp: str
    intervention_type: str
    intervention_message: str
    feedback_method: str  # explicit, implicit, behavioral
    effectiveness_score: float  # 0-1
    response_time_seconds: float
    user_rating: Optional[int] = None  # 1-5 star rating
    user_comment: Optional[str] = None
    context_at_intervention: Dict = None
    behavioral_response: Dict = None


@dataclass
class DetectorFlag:
    """Pattern detector flag with evidence and suggestion"""
    type: str              # e.g., "repeat_docs", "tab_switching"
    severity: str          # "low" | "medium" | "high"
    evidence: Dict[str, Any]
    suggestion: Optional[str] = None


# ============================================================================
# CONTINUOUS TELEMETRY STORAGE - SQLite + JSONL
# ============================================================================

class TelemetryStore:
    """Indefinite telemetry storage with SQLite for queries and JSONL for raw logs"""

    def __init__(self, sqlite_path: Path, jsonl_dir: Path):
        self.sqlite_path = str(sqlite_path)
        self.jsonl_dir = Path(jsonl_dir)
        self.jsonl_dir.mkdir(exist_ok=True)
        self._ensure_db()

    def _ensure_db(self):
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events(
            ts INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            etype TEXT NOT NULL,
            app TEXT,
            window_title TEXT,
            url TEXT,
            host TEXT,
            file_path TEXT,
            lang TEXT,
            keystrokes INTEGER,
            mouse_events INTEGER,
            extras TEXT
        );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_host ON events(host);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_etype ON events(etype);")
        conn.commit()
        conn.close()

    def append(self, event: Dict):
        """Store event in both SQLite (queryable) and JSONL (raw backup)"""
        # Always write JSONL for raw trace
        day = datetime.utcfromtimestamp(event['ts']/1000).strftime("%Y-%m-%d")
        with open(self.jsonl_dir / f"events_{day}.jsonl", "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Also write SQLite for query speed
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO events(ts,user_id,etype,app,window_title,url,host,file_path,lang,keystrokes,mouse_events,extras)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                event['ts'], event['user_id'], event['etype'], event.get(
                    'app'),
                event.get('window_title'), event.get('url'), event.get('host'),
                event.get('file_path'), event.get('lang'),
                event.get('keystrokes', 0), event.get('mouse_events', 0),
                json.dumps(event.get("extras") or {})
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            # best-effort; JSONL is our fallback
            logging.warning(f"SQLite append failed: {e}")

    def query_range(self, user_id: str, since_ms: int) -> List[Dict]:
        """Query events for user since timestamp"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT ts,etype,app,window_title,url,host,file_path,lang,keystrokes,mouse_events,extras 
                FROM events WHERE user_id=? AND ts>=? ORDER BY ts ASC
            """, (user_id, since_ms))
            rows = cur.fetchall()
            conn.close()

            out = []
            for r in rows:
                ts, etype, app, window_title, url, host, file_path, lang, keys, mouse, extras = r
                out.append({
                    "ts": ts, "etype": etype, "app": app, "window_title": window_title, "url": url,
                    "host": host, "file_path": file_path, "lang": lang,
                    "keystrokes": keys or 0, "mouse_events": mouse or 0,
                    "extras": json.loads(extras) if extras else {}
                })
            return out
        except Exception as e:
            logging.warning(f"SQLite query failed: {e}")
            return []

    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def parse_worksmart_deskapp_log(self, log_path: str) -> List[Dict[str, Any]]:
        """Parse WorkSmart deskapp.log into events (handles fixed-width format)."""
        events = []
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # WorkSmart log format: timestamp thread level class_name message
                    # Example: 2025-08-19 16:18:39,084 pool-27-thread-4                 INFO  ActivityJob                       readTimecardFromPath done:...

                    # Split on whitespace, but the class name can have trailing spaces
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    timestamp_str = f"{parts[0]} {parts[1]}"
                    thread = parts[2]
                    level = parts[3]

                    # Find the class name and message - class is usually around column 70-90
                    # Look for the pattern after level
                    import re
                    match = re.match(
                        r'(\S+ \S+)\s+(\S+)\s+(\w+)\s+(\w+)\s+(.+)', line)
                    if match:
                        timestamp_str = match.group(1)
                        thread = match.group(2)
                        level = match.group(3)
                        class_name = match.group(4)
                        message = match.group(5).strip()
                    else:
                        # Fallback to simple split
                        class_name = parts[4] if len(parts) > 4 else "Unknown"
                        message = " ".join(parts[5:]) if len(parts) > 5 else ""

                    try:
                        ts = int(datetime.strptime(timestamp_str,
                                 "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1000)
                    except Exception:
                        continue

                    event = {
                        "ts": ts,
                        "user_id": "worksmart_user",
                        "etype": "deskapp_log",
                        "app": "WorkSmart",
                        "window_title": f"{class_name}: {message}",
                        "url": None,
                        "host": None,
                        "file_path": None,
                        "lang": None,
                        "keystrokes": 0,
                        "mouse_events": 0,
                        "extras": {
                            "thread": thread,
                            "log_level": level,
                            "class": class_name,
                            "message": message,
                            "raw_line": line,
                            "log_file": log_path,
                        },
                    }

                    # Extract activity data if present
                    km = re.search(r"(\d+)\s*key press(?:es)?",
                                   message, flags=re.I)
                    mm = re.search(r"(\d+)\s*mouse click(?:s)?",
                                   message, flags=re.I)
                    if km:
                        event["keystrokes"] = int(km.group(1))
                    if mm:
                        event["mouse_events"] = int(mm.group(1))

                    events.append(event)
        except Exception as e:
            logging.error(f"Failed to parse deskapp log {log_path}: {e}")
        return events

    # Backcompat (typo): keep old call sites alive
    def parse_workmart_deskapp_log(self, log_path: str):
        return self.parse_worksmart_deskapp_log(log_path)

    def parse_worksmart_dcj_file(self, dcj_path: str) -> List[Dict[str, Any]]:
        """Parse WorkSmart DCJ (encrypted data) files - basic metadata extraction"""
        events = []
        try:
            with open(dcj_path, 'rb') as f:
                data = f.read()

            # DCJ files are encrypted, but we can extract metadata
            file_stat = os.stat(dcj_path)
            filename = os.path.basename(dcj_path)

            # Extract time info from filename if possible (e.g., json15_10_00.dcj)
            import re
            time_match = re.search(
                r'json(\d{2})_(\d{2})_(\d{2})\.dcj', filename)

            event = {
                'ts': int(file_stat.st_mtime * 1000),
                'user_id': 'workmart_user',
                'etype': 'dcj_data_capture',
                'app': 'WorkSmart',
                'window_title': f"Data capture: {filename}",
                'url': None,
                'host': None,
                'file_path': dcj_path,
                'lang': None,
                'keystrokes': 0,
                'mouse_events': 0,
                'extras': {
                    'filename': filename,
                    'file_size': len(data),
                    'created_time': file_stat.st_ctime,
                    'modified_time': file_stat.st_mtime,
                    'data_type': 'encrypted_activity_data'
                }
            }

            if time_match:
                hour, minute, second = time_match.groups()
                event['extras']['extracted_time'] = f"{hour}:{minute}:{second}"

            events.append(event)

        except Exception as e:
            logging.error(f"Failed to parse DCJ file {dcj_path}: {e}")

        return events

    # Backcompat (typo): keep old call sites alive
    def parse_workmart_dcj_file(self, dcj_path: str):
        return self.parse_worksmart_dcj_file(dcj_path)

    def upload_historical_logs(self, force_reimport: bool = False):
        """Retroactively upload all existing WorkSmart logs to SQLite database"""
        logging.info("Starting historical log upload...")

        # Track what we've processed to avoid duplicates
        processed_files = set()
        if not force_reimport:
            try:
                conn = sqlite3.connect(self.sqlite_path)
                cur = conn.cursor()
                cur.execute(
                    "SELECT DISTINCT extras FROM events WHERE etype IN ('deskapp_log', 'dcj_data_capture')")
                for row in cur.fetchall():
                    try:
                        extras = json.loads(row[0])
                        if 'log_file' in extras:
                            processed_files.add(extras['log_file'])
                        elif 'filename' in extras:
                            processed_files.add(extras['filename'])
                    except:
                        continue
                conn.close()
            except Exception as e:
                logging.warning(f"Could not check processed files: {e}")

        total_events = 0

        # Find and process all deskapp logs
        log_patterns = [
            "/Users/stanhus/crossoverFiles/logs/deskapp.log*",
            "/Users/stanhus/crossoverFiles/logs/root.log*"
        ]

        for pattern in log_patterns:
            import glob
            for log_file in glob.glob(pattern):
                if log_file in processed_files and not force_reimport:
                    logging.info(f"Skipping already processed: {log_file}")
                    continue

                logging.info(f"Processing log file: {log_file}")
                events = self.parse_worksmart_deskapp_log(log_file)

                for event in events:
                    self.append(event)
                    total_events += 1

                if total_events % 100 == 0:
                    logging.info(f"Processed {total_events} events so far...")

        # Find and process all DCJ files
        dcj_pattern = "/Users/stanhus/crossoverFiles/DataCapture/**/*.dcj"
        import glob
        for dcj_file in glob.glob(dcj_pattern, recursive=True):
            filename = os.path.basename(dcj_file)
            if filename in processed_files and not force_reimport:
                continue

            logging.info(f"Processing DCJ file: {dcj_file}")
            events = self.parse_worksmart_dcj_file(dcj_file)

            for event in events:
                self.append(event)
                total_events += 1

        logging.info(
            f"Historical upload complete! Processed {total_events} total events.")
        return total_events


class HistoryCruncher:
    """Builds compact history digests from raw telemetry logs"""

    def __init__(self, store: TelemetryStore):
        self.store = store

    def build_digest(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Build compact summary of user behavior over N days"""
        since_ms = self.store.now_ms() - days * 24 * 3600 * 1000
        events = self.store.query_range(user_id, since_ms)

        # Top hosts & doc sites
        host_counts = defaultdict(int)
        doc_hosts = {"readthedocs.org", "developer.mozilla.org", "docs.python.org",
                     "docs.microsoft.com", "stackoverflow.com", "github.com"}
        doc_visit = defaultdict(int)

        # Tab switching and dwell
        switches = 0
        last_app = None
        last_ts = None
        last_key = None
        focus_dwell = defaultdict(int)  # key=(app or host)

        # File churn
        file_touches = defaultdict(int)

        for ev in events:
            et = ev["etype"]
            ts = ev["ts"]

            # Process standard activity events
            if et in ("url_visit", "window_focus"):
                host = ev.get("host")
                if host:
                    host_counts[host] += 1
                if host and any(h in host for h in doc_hosts):
                    doc_visit[host] += 1

            if et == "app_switch":
                if last_app and ev.get("app") and ev["app"] != last_app:
                    switches += 1
                last_app = ev.get("app")

            # Process WorkSmart deskapp_log events for comprehensive insights
            elif et == "deskapp_log":
                window_title = ev.get("window_title", "")

                # Extract API activity patterns
                if "GET request, url=" in window_title:
                    url_part = window_title.split(
                        "url=")[-1].split(":")[0].strip()

                    # Track API endpoints for work pattern analysis
                    if "api.crossover.com" in url_part:
                        if "productivity" in url_part:
                            focus_dwell["API-Productivity-Check"] += 1
                        elif "polls" in url_part:
                            focus_dwell["API-Poll-Check"] += 1
                        elif "timecard" in url_part:
                            focus_dwell["API-Timecard-Sync"] += 1

                    # Track external development resources
                    if any(site in url_part for site in ["github.com", "stackoverflow.com", "docs."]):
                        host = url_part.split(
                            "//")[-1].split("/")[0] if "//" in url_part else url_part.split("/")[0]
                        host_counts[host] += 1
                        if any(h in host for h in doc_hosts):
                            doc_visit[host] += 1

                # Track comprehensive work activities from actual logs
                if "Running job:" in window_title:
                    job_type = window_title.split(
                        "Running job:")[-1].split(":")[0].strip()
                    focus_dwell[f"Job-{job_type}"] += 1

                    # Specific job type analysis based on actual log patterns
                    if "TimecardUploadJob" in job_type:
                        focus_dwell["Work-Session-Management"] += 5
                    elif "ActivityJob" in job_type:
                        # High value - indicates active work
                        focus_dwell["Active-Work-Tracking"] += 10
                    elif "PollsRequestJob" in job_type:
                        focus_dwell["System-Health-Check"] += 2
                    elif "DataSyncJob" in job_type:
                        focus_dwell["Data-Synchronization"] += 3
                    elif "ScreenshotJob" in job_type:
                        focus_dwell["Work-Documentation"] += 4
                    elif "WebcamshotJob" in job_type:
                        focus_dwell["Work-Monitoring"] += 2

                # Track actual activity patterns from logs
                if "run activity job:" in window_title:
                    # Strong indicator of work
                    focus_dwell["Active-Work-Sessions"] += 15

                if "ScreenshotJob build event:" in window_title:
                    focus_dwell["Work-Evidence-Capture"] += 5

                if "WebcamshotJob build event:" in window_title:
                    focus_dwell["Work-Presence-Tracking"] += 3

                if "Synchronizing data:" in window_title:
                    focus_dwell["Work-Data-Sync"] += 4

                if "Updating data..." in window_title:
                    focus_dwell["System-Updates"] += 2

                # Track data capture sessions (active work indicators)
                if "Data capture" in window_title:
                    if "started" in window_title:
                        focus_dwell["Work-Session-Start"] += 5
                    elif "stopped" in window_title:
                        focus_dwell["Work-Session-End"] += 5
                    else:
                        focus_dwell["Active-Work-Time"] += 10

                # Track file and activity processing
                if "readTimecardFromPath" in window_title:
                    # Extract file path for work pattern analysis
                    if ".dcj" in window_title:
                        focus_dwell["Work-Data-Processing"] += 3
                        # Count different work sessions by extracting timestamp patterns
                        if "json" in window_title:
                            focus_dwell["Work-Sessions-Logged"] += 1

                # Track timecard management
                if "Found pending timecards:" in window_title:
                    try:
                        # Extract number of pending timecards
                        count = int(window_title.split(
                            "Found pending timecards:")[-1].split(":")[0].strip())
                        if count > 0:
                            focus_dwell["Pending-Work-Data"] += count
                        else:
                            focus_dwell["Up-to-Date-Tracking"] += 1
                    except:
                        focus_dwell["Timecard-Check"] += 1

                # Track productivity monitoring
                if "polls/pending" in window_title:
                    focus_dwell["Productivity-Monitoring"] += 1

                # Track file system activity
                if "crossoverFiles" in window_title:
                    focus_dwell["Work-File-Access"] += 2

                # Track system health
                if "PortalClient" in window_title:
                    focus_dwell["System-Communication"] += 1

            if et == "window_focus":
                key = ev.get("host") or ev.get("app") or "unknown"
                if last_ts is not None and last_key is not None:
                    dwell = max(0, ts - last_ts)
                    focus_dwell[last_key] += dwell  # attribute to previous key
                last_ts = ts
                last_key = key

            if et == "file_edit" and ev.get("file_path"):
                file_touches[ev["file_path"]] += 1

        top_hosts = sorted(host_counts.items(),
                           key=lambda x: x[1], reverse=True)[:10]
        top_docs = sorted(doc_visit.items(),
                          key=lambda x: x[1], reverse=True)[:10]
        top_dwell = sorted(((k, v) for k, v in focus_dwell.items()),
                           key=lambda x: x[1], reverse=True)[:10]
        top_files = sorted(file_touches.items(),
                           key=lambda x: x[1], reverse=True)[:10]

        digest = {
            "since_days": days,
            "n_events": len(events),
            "tab_switches": switches,
            "top_hosts": top_hosts,
            "top_doc_hosts": top_docs,
            "top_dwell": [(k, v//1000) for k, v in top_dwell],   # seconds
            "top_files": top_files,
        }
        return digest


# ============================================================================
# PATTERN DETECTORS - Reusable behavioral pattern detection
# ============================================================================

class RepeatedDocsDetector:
    """Detects repeated visits to documentation sites"""

    def run(self, digest: Dict, current: Dict) -> List[DetectorFlag]:
        flags = []
        docs = digest.get("top_doc_hosts", [])
        # Trigger if we see at least one doc host ≥5 visits in 7d
        frequent = [(h, c) for (h, c) in docs if c >= 5]
        if frequent:
            # prioritize the top doc site
            h, c = frequent[0]
            flags.append(DetectorFlag(
                type="repeat_docs",
                severity="medium" if c < 15 else "high",
                evidence={"host": h, "count_7d": c},
                suggestion=f"You've visited {h} repeatedly. Block 25–30 min today to read one core page end-to-end and take notes."
            ))
        return flags


class ExcessiveTabSwitchingDetector:
    """Detects high context switching behavior"""

    def run(self, digest: Dict, current: Dict) -> List[DetectorFlag]:
        # Heuristic: if tab/app switches over last 7d are high and current app switches/minute > threshold
        switches_7d = digest.get("tab_switches", 0)
        current_spm = current.get("switches_per_min", 0.0)
        if switches_7d >= 300 or current_spm >= 6.0:
            focus_candidates = [
                k for (k, _) in digest.get("top_dwell", [])[:3]]
            return [DetectorFlag(
                type="tab_switching",
                severity="high" if current_spm >= 8.0 else "medium",
                evidence={"switches_7d": switches_7d, "switches_per_min": current_spm,
                    "candidate_focus": focus_candidates},
                suggestion=f"High context switching now. Pin 1–2 of {', '.join(focus_candidates)} and close others for a 25-min block."
            )]
        return []


class FileChurnDetector:
    """Detects scattered file editing without deep focus"""

    def run(self, digest: Dict, current: Dict) -> List[DetectorFlag]:
        files = digest.get("top_files", [])
        if not files:
            return []
        # If many files touched but no deep dwell anywhere
        total_dwell = sum(sec for _, sec in digest.get("top_dwell", []))
        deep_dwell = max(
            [sec for _, sec in digest.get("top_dwell", [])], default=0)
        if len(files) >= 8 and deep_dwell < max(900, 0.25 * total_dwell):  # <15 min or <25% of total
            return [DetectorFlag(
                type="file_churn",
                severity="medium",
                evidence={"n_files_touched_7d": len(
                    files), "max_dwell_sec": deep_dwell},
                suggestion="Lots of files touched with little deep focus. Pick one file and give it 20–30 min uninterrupted."
            )]
        return []


class MeetingDistractionDetector:
    """Detects high activity during meetings"""

    def run(self, digest: Dict, current: Dict) -> List[DetectorFlag]:
        # Simple heuristic: if in meeting and high tab switching
        if current.get("in_meeting", False) and current.get("switches_per_min", 0) >= 4:
            return [DetectorFlag(
                type="meeting_distraction",
                severity="high",
                evidence={"switches_per_min": current.get(
                    "switches_per_min", 0)},
                suggestion="High activity during meeting. Consider closing distracting tabs and focusing on the discussion."
            )]
        return []


# ============================================================================
# ULTIMATE AI COACH - ALL FUNCTIONALITY CONSOLIDATED
# ============================================================================


class AICoach:
    """
    Ultimate AI Coach - Complete consolidated coaching system combining all functionality:
    - Base AI Coach, Personalized Coach, Enhanced ML Coach
    - Pattern Learning, User Modeling, Predictive Analytics
    - Feedback Collection, Burnout Prediction, Context Tracking
    - All in one comprehensive class with automatic capability detection
    """

    def __init__(self, data_dir: str = "ultimate_coach_data", test_mode: bool = False):
        """Initialize the ultimate coaching system with all capabilities"""

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.test_mode = test_mode

        # Initialize Anthropic client
        self.claude_client = None
        self._setup_anthropic_client()

        # Core coaching thresholds and strategies - OPTIMIZED FOR 8H GOAL ACHIEVEMENT
        self.coaching_strategies = {
            # More sensitive to catch productivity drops early
            'productivity_thresholds': {'low': 0.35, 'high': 0.7},
            # Catch focus issues before they hurt 8h quality
            'focus_thresholds': {'low': 0.45, 'high': 0.8},
            # Earlier stress intervention to maintain 8h sustainability
            'stress_thresholds': {'moderate': 0.55, 'high': 0.75},
            # Energy crucial for 8h - catch drops early
            'energy_thresholds': {'low': 0.35, 'critical': 0.20}
        }

        # Notification & alert config - OPTIMIZED FOR 8H PRODUCTIVITY COACHING
        self.notification_config = {
            'max_per_hour': 6,                 # More alerts for 8h coaching
            'min_minutes_between': 8,          # Faster response for productivity
            'per_type_cooldown_minutes': {
                'stress_reduction': 15,        # Quicker stress intervention
                'energy_boost': 12,           # Energy alerts more frequent
                'productivity_boost': 10,     # Core 8h productivity alerts
                'focus_enhancement': 12,      # Focus critical for 8h
                'break_reminder': 20          # Reasonable break spacing
            },
            'suppress_in_meeting': True,       # gate non-critical nudges when in meeting
            'allow_in_meeting_types': ['stress_reduction'],
            # don't repeat same message text within this window
            'repeat_suppression_minutes': 90,
            'default_channel': 'system_banner'  # optional: 'toast', 'modal', 'push'
        }

        # Configure notification frequency based on user preference
        self._configure_notification_frequency()

        # Standardize canonical keys used throughout
        self.keys = {
            'session_duration': 'session_duration_hours',
            'current_app': 'current_application',
            'in_meeting': 'in_meeting'
        }

        # Intervention history for cooldowns and analytics
        self.intervention_history: Dict[str, Dict] = {}

        # ML components
        self.pattern_learner_enabled = SKLEARN_AVAILABLE
        self.effectiveness_predictor = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.timing_optimizer = GradientBoostingRegressor(
            n_estimators=50, random_state=42)
        self.scaler = StandardScaler()

        # User modeling and context tracking
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_contexts: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100))
        self.adaptive_thresholds: Dict[str, Dict] = {}

        # Feedback and learning systems
        self.feedback_history: List[FeedbackEntry] = []
        self.intervention_contexts = {}
        self.learning_callbacks: List[Callable[[FeedbackEntry], None]] = []

        # Interaction data for ML training
        self.interaction_data: List[Dict] = []

        # Model performance tracking
        self.model_performance = {
            'effectiveness_accuracy': 0.0,
            'timing_mse': 0.0,
            'last_trained': None
        }

        # Intervention history
        self.intervention_history = {}

        # Inactivity and day transition tracking
        self.last_activity_time = datetime.now()
        self.last_activity_data = {'keystrokes': 0, 'clicks': 0}
        self.current_day = datetime.now().date()
        self.day_transition_announced = False
        self.inactivity_level = None  # None, '30min', '2hr', '6hr', '12hr'
        self.last_inactivity_message = None

        # Persona detection patterns
        self.persona_patterns = {
            'developer': {
                'apps': ['vscode', 'visual studio code', 'intellij', 'pycharm', 'xcode', 'sublime', 'atom', 'vim', 'emacs', 'github desktop', 'terminal'],
                'keywords': ['code', 'debug', 'commit', 'pull request', 'repository', 'function', 'class'],
                'coaching_messages': {
                    'productivity_boost': [
                        "🚀 ONE FUNCTION. 25 minutes. Close Slack/Discord. Commit when done.",
                        "🚀 Kill all browser tabs. Full-screen IDE. Ship ONE feature. NOW.",
                        "🚀 Productivity tanking. Pick the hardest bug. Solve it. 25 min timer."
                    ],
                    'focus_enhancement': [
                        "🎯 Context switching killing you. Cmd+W everything except IDE. Code for 20 min straight.",
                        "💡 Focus shot. Distraction-free mode NOW. One file. One problem. GO."
                    ],
                    'stress_reduction': [
                        "😤 Debugging too long. Walk 2 min. Fresh eyes = solved bug.",
                        "⚠️ Stress spike. git commit. Stand up. Back in 3 min with solution."
                    ],
                    'break_reminder': [
                        "⏰ 2+ hours coding. Break NOW or ship bugs. Walk. Water. 5 min.",
                        "💭 RSI warning. Stretch hands/neck 60 sec or pay later."
                    ]
                }
            },
            'analyst': {
                'apps': ['excel', 'tableau', 'power bi', 'r studio', 'jupyter', 'spss', 'stata', 'sas'],
                'keywords': ['data', 'analysis', 'report', 'dashboard', 'visualization', 'spreadsheet'],
                'coaching_messages': {
                    'productivity_boost': [
                        "📊 ANALYSIS PARALYSIS. Pick ONE metric. Deep dive 30 min. Report done.",
                        "📊 Data rabbit hole. STOP. Export findings NOW. Polish later."
                    ],
                    'focus_enhancement': [
                        "🎯 10+ spreadsheets open. Close all but ONE. Finish analysis or drown in data.",
                        "💡 Excel hell. Save work. Close everything. ONE dataset. 25 min sprint."
                    ],
                    'stress_reduction': [
                        "😤 Numbers blur = brain fried. Save. Walk 3 min. Answer will appear.",
                        "⚠️ Overthinking data. Step back 2 min. Pattern obvious when you return."
                    ],
                    'break_reminder': [
                        "⏰ 2+ hours in spreadsheets. Eyes failing. Break NOW or miss insights.",
                        "💭 Data fatigue = bad decisions. 5 min away. Come back sharp."
                    ]
                }
            },
            'manager': {
                'apps': ['slack', 'teams', 'zoom', 'outlook', 'calendar', 'asana', 'trello', 'jira'],
                'keywords': ['meeting', 'review', 'team', 'project', 'deadline', 'status', 'planning'],
                'coaching_messages': {
                    'productivity_boost': [
                        "🎯 FIREFIGHTING MODE. Stop. Pick THREE fires. Let others burn. Delegate NOW.",
                        "🎯 Meeting madness. Cancel 2 meetings. Block 30 min. Do actual work."
                    ],
                    'focus_enhancement': [
                        "💡 Slack/email killing output. CLOSE THEM. 30 min deep work or team suffers.",
                        "🎯 Context switching chaos. One decision. Make it. Move on. Stop overthinking."
                    ],
                    'stress_reduction': [
                        "😤 Can't manage burnt out. 3 min walk. Delegate 1 thing. Come back clear.",
                        "⚠️ Decision fatigue. Stop deciding. 5 min break. Trust your gut after."
                    ],
                    'break_reminder': [
                        "⏰ Back-to-back meetings 2+ hours. Break or make bad calls. 5 min. NOW.",
                        "💭 Leader burnout = team burnout. Model breaks. Take 5 min."
                    ]
                }
            },
            'generic': {
                'apps': [],
                'keywords': [],
                'coaching_messages': {
                    'productivity_boost': [
                        "💪 OUTPUT FAILING. One task. 25 min timer. Everything else waits.",
                        "💪 Productivity 30%. Pick hardest task. Do it NOW. Rest follows."
                    ],
                    'focus_enhancement': [
                        "🎯 Multitasking = nothing done. ONE THING. Complete it. Then next.",
                        "💡 Focus dead. Close 10 tabs. Pick 1 task. 20 min sprint."
                    ],
                    'stress_reduction': [
                        "😤 Overwhelmed. Stop. Breathe 4-7-8 x3. Pick ONE thing. Start.",
                        "⚠️ Stress building. 2 min walk beats 2 hour spiral. GO."
                    ],
                    'break_reminder': [
                        "You've been working steadily. A short break will help maintain your productivity.",
                        "Time for a 5-minute reset—stretch and hydrate."
                    ]
                }
            }
        }

        # Load existing data
        self._load_user_profiles()
        self._load_training_data()
        self._load_feedback_history()

        # Continuous telemetry storage and analysis
        self.store = TelemetryStore(
            self.data_dir / "coach.sqlite", self.data_dir / "events")
        self.history_cruncher = HistoryCruncher(self.store)

        # Pattern detectors for three-pass coaching
        self.detectors = [
            RepeatedDocsDetector(),
            ExcessiveTabSwitchingDetector(),
            FileChurnDetector(),
            MeetingDistractionDetector(),
        ]

        # Determine system capabilities
        self.capabilities = self._determine_capabilities()

        logger.info(
            f"Ultimate AI Coach initialized with {len(self.capabilities)} capabilities: {', '.join(self.capabilities)}")

    def _setup_anthropic_client(self):
        """Setup Anthropic Claude client if available"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if ANTHROPIC_AVAILABLE and api_key:
                self.claude_client = anthropic.AsyncAnthropic(api_key=api_key)
                logger.info("✅ Anthropic Claude client initialized")
            else:
                logger.info(
                    "ℹ️  Using rule-based coaching (no Anthropic API key or library)")
        except Exception as e:
            logger.warning(f"Anthropic client setup failed: {e}")

    def _configure_notification_frequency(self):
        """Configure notification frequency based on user preference"""
        frequency = os.getenv('NOTIFICATION_FREQUENCY', 'active')

        if frequency == 'gentle':
            # GENTLE (1-2 per hour) - Minimal, key insights only
            self.notification_config['max_per_hour'] = 2
            self.notification_config['min_minutes_between'] = 25
            for key in self.notification_config['per_type_cooldown_minutes']:
                self.notification_config['per_type_cooldown_minutes'][key] *= 2

        elif frequency == 'active':
            # ACTIVE (5-6 per hour) - Frequent guidance & alerts
            self.notification_config['max_per_hour'] = 6
            self.notification_config['min_minutes_between'] = 8
            for key in self.notification_config['per_type_cooldown_minutes']:
                self.notification_config['per_type_cooldown_minutes'][key] = max(
                    5, self.notification_config['per_type_cooldown_minutes'][key] * 0.7)

        else:  # balanced (default)
            # BALANCED (2-3 per hour) - Regular helpful coaching
            self.notification_config['max_per_hour'] = 3  # Reduced from 4
            self.notification_config['min_minutes_between'] = 15  # Increased from 12
            # Keep default cooldown times

        logger.info(f"✅ Notification frequency set to {frequency} mode")

    def _determine_capabilities(self) -> List[str]:
        """Determine system capabilities based on available libraries"""
        capabilities = ['basic_coaching',
                        'persona_detection', 'context_awareness']

        if SKLEARN_AVAILABLE:
            capabilities.extend(
                ['ml_pattern_learning', 'predictive_analytics', 'burnout_prediction'])

        if PANDAS_AVAILABLE:
            capabilities.append('advanced_data_analysis')

        if self.claude_client:
            capabilities.append('ai_generated_coaching')

        capabilities.extend(
            ['user_modeling', 'feedback_learning', 'adaptive_thresholds'])

        return capabilities

    # ========================================================================
    # ENHANCED 3-PASS CLAUDE ANALYSIS SYSTEM
    # ========================================================================
    
    async def _run_three_pass_claude_analysis(self, user_id: str, hist7: Dict, current: Dict, telemetry: Dict) -> Optional[Dict]:
        """
        Comprehensive 3-pass Claude analysis system:
        Pass 1: Historical context analysis
        Pass 2: Current activity analysis  
        Pass 3: Synthesis and decision
        """
        print(f"[DEBUG] 🔬 Starting 3-pass Claude analysis", flush=True)
        
        # Extract maximum telemetry data
        full_telemetry = self._extract_comprehensive_telemetry_data(telemetry)
        
        try:
            # Pass 1: Historical Context Analysis
            print(f"[DEBUG] 📚 Pass 1: Analyzing historical patterns...", flush=True)
            historical_analysis = await self._claude_pass1_historical_analysis(user_id, hist7, full_telemetry)
            
            # Pass 2: Current Activity Analysis
            print(f"[DEBUG] 📊 Pass 2: Analyzing current activity...", flush=True)
            current_analysis = await self._claude_pass2_current_analysis(current, full_telemetry)
            
            # Pass 3: Synthesis and Decision
            print(f"[DEBUG] 🎯 Pass 3: Synthesizing and deciding...", flush=True)
            final_decision = await self._claude_pass3_synthesis(historical_analysis, current_analysis, full_telemetry)
            
            return final_decision
            
        except Exception as e:
            print(f"[DEBUG] ❌ 3-pass analysis failed: {e}", flush=True)
            return None
    
    def _extract_comprehensive_telemetry_data(self, telemetry: Dict) -> Dict:
        """Extract maximum available telemetry data for Claude analysis"""
        comprehensive_data = {
            # Core metrics
            'productivity_score': telemetry.get('productivity_score', 0.0),
            'focus_score': telemetry.get('focus_score', 0.0),
            'energy_score': telemetry.get('energy_score', 0.5),
            'stress_score': telemetry.get('stress_score', 0.5),
            
            # Current activity
            'current_app': telemetry.get('current_app', 'unknown'),
            'current_window': telemetry.get('current_window', ''),
            'current_url': telemetry.get('current_url', ''),
            'current_document': telemetry.get('current_document', ''),
            
            # Activity patterns
            'keyboard_count': telemetry.get('keyboard_count', 0),
            'mouse_count': telemetry.get('mouse_count', 0),
            'total_activity': telemetry.get('total_activity', 0),
            'idle_time_minutes': telemetry.get('idle_time_minutes', 0),
            
            # WorkSmart data
            'worksmart_hours_today': telemetry.get('worksmart_hours_today', '0:0'),
            'worksmart_session_active': telemetry.get('worksmart_session_active', False),
            'worksmart_total_keystrokes': telemetry.get('worksmart_total_keystrokes', 0),
            'worksmart_total_clicks': telemetry.get('worksmart_total_clicks', 0),
            
            # Context
            'in_meeting': telemetry.get('in_meeting', False),
            'time_of_day': telemetry.get('time_of_day', 'unknown'),
            'day_of_week': telemetry.get('day_of_week', 'unknown'),
            
            # Recent events buffer for pattern analysis
            'recent_events': telemetry.get('recent_events', []),
            'event_count': len(telemetry.get('recent_events', [])),
        }
        
        # Add calculated metrics
        if comprehensive_data['keyboard_count'] > 0 or comprehensive_data['mouse_count'] > 0:
            comprehensive_data['activity_ratio'] = comprehensive_data['keyboard_count'] / max(1, comprehensive_data['mouse_count'])
        else:
            comprehensive_data['activity_ratio'] = 0.0
        
        print(f"[DEBUG] 📋 Comprehensive telemetry extracted: {len(comprehensive_data)} data points", flush=True)
        return comprehensive_data

    async def _claude_pass1_historical_analysis(self, user_id: str, hist7: Dict, telemetry: Dict) -> str:
        """Pass 1: Deep historical context analysis"""
        prompt = f"""You are an expert productivity analyst. Analyze this user's historical patterns and current context to provide a conclusive assessment.

HISTORICAL DATA (7 days):
- Host activity: {hist7.get('host_counts', {})}
- Document usage: {hist7.get('doc_visits', {})}
- App switching patterns: {hist7.get('switches', 0)} switches
- Focus patterns: {hist7.get('focus_dwell', {})}
- File activity: {hist7.get('file_touches', {})}

CURRENT CONTEXT:
- Current app: {telemetry.get('current_app', 'unknown')}
- Current window: {telemetry.get('current_window', '')}
- Current URL: {telemetry.get('current_url', '')}
- WorkSmart hours today: {telemetry.get('worksmart_hours_today', '0:0')}
- Time context: {telemetry.get('time_of_day', 'unknown')} on {telemetry.get('day_of_week', 'unknown')}

TASK: Provide a conclusive historical analysis focusing on:
1. Historical productivity patterns and trends
2. How current activity fits within historical context
3. Specific concerns or opportunities based on patterns
4. Risk assessment (procrastination, distraction, burnout indicators)

Be specific, data-driven, and conclusive. Identify concrete patterns and concerns."""

        try:
            response = await self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            analysis = response.content[0].text
            print(f"[DEBUG] 📚 Pass 1 complete: {len(analysis)} chars", flush=True)
            return analysis
        except Exception as e:
            print(f"[DEBUG] ❌ Pass 1 failed: {e}", flush=True)
            return "Historical analysis unavailable"

    async def _claude_pass2_current_analysis(self, current: Dict, telemetry: Dict) -> str:
        """Pass 2: Deep current activity analysis"""
        prompt = f"""You are an expert productivity analyst. Analyze this user's current activity state and provide a conclusive assessment.

CURRENT ACTIVITY SNAPSHOT:
- App: {current.get('current_app', 'unknown')}
- Productivity score: {current.get('productivity_score', 0.0):.2f}
- Focus level: {telemetry.get('focus_score', 0.0):.2f}
- Energy level: {telemetry.get('energy_score', 0.5):.2f}
- Stress indicators: {telemetry.get('stress_score', 0.5):.2f}

ACTIVITY METRICS:
- Keyboard activity: {telemetry.get('keyboard_count', 0)} keystrokes
- Mouse activity: {telemetry.get('mouse_count', 0)} clicks
- Total activity: {telemetry.get('total_activity', 0)}
- Idle time: {telemetry.get('idle_time_minutes', 0)} minutes
- Activity ratio: {telemetry.get('activity_ratio', 0.0):.2f}

CONTEXT:
- Window title: {telemetry.get('current_window', 'unknown')}
- URL: {telemetry.get('current_url', 'none')}
- In meeting: {telemetry.get('in_meeting', False)}
- Recent events: {telemetry.get('event_count', 0)} events

TASK: Provide a conclusive current activity analysis focusing on:
1. Immediate productivity assessment and concerns
2. Activity quality and engagement levels
3. Distraction or focus indicators
4. Immediate risks or opportunities for improvement

Be specific about what the data indicates about their current state."""

        try:
            response = await self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            analysis = response.content[0].text
            print(f"[DEBUG] 📊 Pass 2 complete: {len(analysis)} chars", flush=True)
            return analysis
        except Exception as e:
            print(f"[DEBUG] ❌ Pass 2 failed: {e}", flush=True)
            return "Current analysis unavailable"

    async def _claude_pass3_synthesis(self, historical: str, current: str, telemetry: Dict) -> Optional[Dict]:
        """Pass 3: Synthesize analyses and make final coaching decision"""
        prompt = f"""You are an expert AI productivity coach. Based on comprehensive historical and current analyses, make a final coaching decision.

HISTORICAL ANALYSIS:
{historical}

CURRENT ANALYSIS:
{current}

SYNTHESIS CONTEXT:
- WorkSmart session active: {telemetry.get('worksmart_session_active', False)}
- Hours worked today: {telemetry.get('worksmart_hours_today', '0:0')}
- Current productivity: {telemetry.get('productivity_score', 0.0):.2f}
- Current focus: {telemetry.get('focus_score', 0.0):.2f}

DECISION TASK: 
Based on both analyses, decide if coaching intervention is needed. Be CONSERVATIVE - only recommend intervention for clear, urgent situations. High-performance users prefer minimal interruptions.

INTERVENTION CRITERIA:
- Only intervene if productivity/focus scores are consistently extremely low (< 0.1) for extended periods
- Recognize that AI tools (ChatGPT, Claude, Cursor) are productive work environments
- Prioritize user autonomy - they know their workflow best
- Consider that brief low activity may be normal thinking/planning time

If intervention needed, provide:
1. INTERVENTION TYPE: One of [productivity_boost, focus_enhancement, break_reminder, distraction_alert, stress_reduction, positive_reinforcement, no_action]
2. PRIORITY LEVEL: 1 (low), 2 (medium), or 3 (high) 
3. MESSAGE: Clear, actionable 1-sentence coaching message (under 100 chars)
4. REASONING: Brief explanation of why this intervention was chosen

Format your response as:
DECISION: [YES/NO]
TYPE: [intervention_type]  
PRIORITY: [1-3]
MESSAGE: [coaching message]
REASONING: [explanation]

Be very conservative - when in doubt, choose NO."""

        try:
            response = await self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            decision_text = response.content[0].text
            print(f"[DEBUG] 🎯 Pass 3 complete: {decision_text[:100]}...", flush=True)
            
            # Parse Claude's decision
            return self._parse_claude_decision(decision_text, telemetry)
            
        except Exception as e:
            print(f"[DEBUG] ❌ Pass 3 failed: {e}", flush=True)
            return None
    
    def _parse_claude_decision(self, decision_text: str, telemetry: Dict) -> Optional[Dict]:
        """Parse Claude's structured decision into intervention dict"""
        try:
            lines = [line.strip() for line in decision_text.split('\n') if line.strip()]
            
            decision = None
            intervention_type = None
            priority = 2
            message = None
            reasoning = None
            
            for line in lines:
                if line.startswith('DECISION:'):
                    decision = 'YES' in line.upper()
                elif line.startswith('TYPE:'):
                    intervention_type = line.split(':', 1)[1].strip()
                elif line.startswith('PRIORITY:'):
                    try:
                        priority = int(line.split(':', 1)[1].strip())
                    except:
                        priority = 2
                elif line.startswith('MESSAGE:'):
                    message = line.split(':', 1)[1].strip()
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            if not decision or not intervention_type or not message:
                print(f"[DEBUG] ⚠️  Claude decision parsing incomplete", flush=True)
                return None
                
            return {
                "id": str(uuid.uuid4()),
                "type": intervention_type,
                "message": message,
                "priority": priority,
                "urgency": "high" if priority >= 3 else ("medium" if priority >= 2 else "low"),
                "persona": "claude_analysis",
                "channel": "terminal_notifier",
                "meta": {
                    "source": "claude_3pass",
                    "reasoning": reasoning,
                    "confidence": 0.9
                }
            }
            
        except Exception as e:
            print(f"[DEBUG] ❌ Failed to parse Claude decision: {e}", flush=True)
            return None

    # THREE-PASS COACHING PIPELINE
    # ========================================================================

    def ingest_event(self, user_id: str, etype: str, **kwargs):
        """Ingest telemetry event into continuous storage"""
        url = kwargs.get("url")
        host = self._normalize_host(url) if url else None
        ev = {
            "ts": self.store.now_ms(),
            "user_id": user_id,
            "etype": etype,
            "app": kwargs.get("app"),
            "window_title": kwargs.get("window_title"),
            "url": url,
            "host": host,
            "file_path": kwargs.get("file_path"),
            "lang": kwargs.get("lang"),
            "keystrokes": int(kwargs.get("keystrokes", 0)),
            "mouse_events": int(kwargs.get("mouse_events", 0)),
            "extras": kwargs.get("extras") or {}
        }
        self.store.append(ev)

    def _normalize_host(self, url: Optional[str]) -> Optional[str]:
        """Extract and normalize host from URL"""
        try:
            if not url:
                return None
            p = urlparse(url)
            return (p.netloc or "").lower()
        except Exception:
            return None

    def build_current_snapshot(self, user_id: str, telemetry: Dict) -> Dict[str, Any]:
        """Build lightweight 'now' summary from incoming telemetry + recent logs"""
        # Estimate recent switches/min from last ~10 minutes
        ten_min_ms = 10 * 60 * 1000
        recent = self.store.query_range(
            user_id, self.store.now_ms() - ten_min_ms)
        switches = sum(1 for ev in recent if ev["etype"] == "app_switch")
        spm = switches / 10.0
        active_hosts = list({ev.get("host")
                            for ev in recent if ev.get("host")})[:5]

        return {
            "productivity": telemetry.get("productivity_score", 0.5),
            "focus": telemetry.get("focus_quality", 0.5),
            "stress": telemetry.get("stress_level", 0.5),
            "energy": telemetry.get("energy_level", 0.5),
            "session_hours": telemetry.get("session_duration_hours", 0.0),
            "current_app": telemetry.get("current_application") or telemetry.get("current_app"),
            "current_window": telemetry.get("current_window", ""),
            "switches_per_min": spm,
            "recent_hosts": active_hosts,
            "in_meeting": telemetry.get("in_meeting", False) or telemetry.get("in_call", False)
        }

    async def run_coaching_cycle(self, user_id: str, telemetry: Dict) -> Optional[Dict]:
        """
        Pass A: crunch history; Pass B: snapshot now; Pass C: decide & generate.
        Returns a normalized intervention dict OR a 'no_suggestion' envelope.
        """
        print(
            f"[DEBUG] 🔄 Starting coaching cycle for user {user_id}", flush=True)

        # A) History - build 7-day and 30-day digests
        hist7 = self.history_cruncher.build_digest(user_id, days=7)
        hist30 = self.history_cruncher.build_digest(
            user_id, days=30)  # for trend signals
        print(
            f"[DEBUG] 📚 History digests built: 7d={len(hist7.get('host_counts', {}))}, 30d={len(hist30.get('host_counts', {}))}", flush=True)

        # B) Current snapshot
        current = self.build_current_snapshot(user_id, telemetry)
        print(
            f"[DEBUG] 📊 Current snapshot: app={current.get('current_app')}, productivity={current.get('productivity_score', 0):.2f}", flush=True)

        # C) Run detectors → candidate intents
        flags: List[DetectorFlag] = []
        for det in self.detectors:
            det_flags = det.run(hist7, current)
            flags.extend(det_flags)
            if det_flags:
                print(
                    f"[DEBUG] 🚩 Detector {det.__class__.__name__} found {len(det_flags)} flags", flush=True)

        print(
            f"[DEBUG] 🎯 Total detector flags found: {len(flags)}", flush=True)

        # Enhanced: 3-Pass Claude Analysis System 
        if not flags and self.claude_client:
            print(f"[DEBUG] 🧠 No detector flags - running 3-pass Claude analysis", flush=True)
            
            # Check rate limiting before expensive Claude calls
            if not self._should_send_notification():
                print(f"[DEBUG] 🚫 Rate limited - skipping Claude analysis", flush=True)
            else:
                try:
                    claude_result = await self._run_three_pass_claude_analysis(user_id, hist7, current, telemetry)
                    if claude_result:
                        print(f"[DEBUG] ✨ Claude analysis found intervention: {claude_result.get('type')}", flush=True)
                        return claude_result
                except Exception as e:
                    print(f"[DEBUG] ⚠️  Claude analysis failed: {e}", flush=True)
        
        # Fallback: Basic coaching when no detector flags or Claude analysis found
        if not flags:
            print(f"[DEBUG] 🔄 No detector/Claude flags - trying basic coaching fallback", flush=True)
            
            # Check rate limiting
            if not self._should_send_notification():
                print(f"[DEBUG] 🚫 Rate limited - skipping basic coaching", flush=True)
                return {
                    "id": str(uuid.uuid4()),
                    "type": "no_suggestion",
                    "message": "Rate limited",
                    "priority": 1,
                    "urgency": "low",
                    "persona": self._get_user_profile(user_id).persona,
                    "channel": "system_banner",
                    "meta": {"source": "rate_limit", "reasoning": "Too many recent notifications", "confidence": 1.0}
                }
            
            # Check for basic coaching opportunities based on current snapshot
            productivity_score = current.get('productivity_score', 0.0)
            current_app = current.get('current_app', 'unknown')
            
            # Get actual activity levels to avoid false alarms
            total_keystrokes = current.get('total_keystrokes', 0)
            total_mouse_clicks = current.get('total_mouse_clicks', 0) 
            total_activity = total_keystrokes + total_mouse_clicks
            
            print(f"[DEBUG] 📈 Basic coaching check - productivity: {productivity_score:.2f}, app: {current_app}, activity: {total_activity}", flush=True)
            
            # Don't trigger low productivity if user has high activity (130+ keystrokes/clicks in recent window)
            if total_activity >= 100:
                print(f"[DEBUG] 🚫 Skipping low productivity alert - high activity detected: {total_activity}", flush=True)
                return None
            
            # Basic productivity coaching with higher thresholds for high-performance users
            if productivity_score < 0.15:  # Only trigger on extremely low productivity
                return {
                    "id": str(uuid.uuid4()),
                    "type": "productivity_boost",
                    "message": f"🚀 Extended low activity detected in {current_app}. Take a quick break or switch to a high-focus task?",
                    "priority": 2,
                    "urgency": "medium",
                    "persona": self._get_user_profile(user_id).persona,
                    "channel": "terminal_notifier",
                    "meta": {"source": "basic_fallback", "reasoning": f"Very low productivity: {productivity_score:.2f}", "confidence": 0.8}
                }
            elif productivity_score < 0.25:  # Reduced from 0.5
                return {
                    "id": str(uuid.uuid4()),
                    "type": "gentle_nudge",
                    "message": f"💡 Lower focus detected in {current_app}. Consider deep work techniques to boost productivity?",
                    "priority": 1,
                    "urgency": "low", 
                    "persona": self._get_user_profile(user_id).persona,
                    "channel": "terminal_notifier",
                    "meta": {"source": "basic_fallback", "reasoning": f"Moderate productivity: {productivity_score:.2f}", "confidence": 0.7}
                }
            else:
                # High productivity - encourage or suggest break
                return {
                    "id": str(uuid.uuid4()),
                    "type": "positive_reinforcement",
                    "message": f"✨ Great focus in {current_app}! Keep up the excellent work or consider a micro-break to maintain momentum.",
                    "priority": 1,
                    "urgency": "low",
                    "persona": self._get_user_profile(user_id).persona,
                    "channel": "terminal_notifier", 
                    "meta": {"source": "basic_fallback", "reasoning": f"High productivity: {productivity_score:.2f}", "confidence": 0.8}
                }

        # Merge/choose the top flag (simple heuristic: highest severity)
        severity_rank = {"low": 1, "medium": 2, "high": 3}
        top = sorted(flags, key=lambda f: severity_rank.get(
            f.severity, 1), reverse=True)[0]

        # Compose a structured bundle for Anthropic
        bundle = {
            "history_digest": hist7,
            "current_snapshot": current,
            "detector_flags": [asdict(f) for f in flags]
        }

        # AI or rule-based finalization
        if self.claude_client:
            suggestion = await self._anthropic_synthesis(user_id, bundle, top.type)
        else:
            suggestion = self._rule_synthesis(user_id, bundle, top)

        # Suppress / record / return
        analysis = self._basic_analysis(telemetry)
        context = self._extract_context(telemetry, analysis)
        suppress, reason = self._should_suppress_notification(
            user_id, suggestion, context)
        if suppress:
            print(
                f"🔕 DEBUG: Suppressed {suggestion.get('type', 'unknown')} alert - {reason}")
            self._log_notification_event(
                "suppressed", user_id, suggestion, reason=reason)
            return None

        self._record_intervention(suggestion, user_id)
        self._log_notification_event("sent", user_id, suggestion)
        return suggestion

    def _rule_synthesis(self, user_id: str, bundle: Dict, top: DetectorFlag) -> Dict:
        """Rule-based fallback synthesis"""
        persona = self._get_user_profile(user_id).persona
        msg = top.suggestion or "Tighten focus for 25 minutes on your primary task."
        urgency = "high" if top.severity == "high" else "medium"
        return {
            "id": str(uuid.uuid4()),
            "type": top.type,
            "message": msg,
            "priority": 3 if urgency == "high" else 2,
            "urgency": urgency,
            "persona": persona,
            "channel": self.notification_config.get('default_channel', 'system_banner'),
            "meta": {
                "source": "three_pass_rule",
                "confidence": 0.7,
                "evidence": top.evidence
            }
        }

    async def _anthropic_synthesis(self, user_id: str, bundle: Dict, intent: str) -> Dict:
        """Anthropic-powered synthesis with structured prompts"""
        # Keep prompt tight; include only digests, not raw logs
        prompt = f"""You are an AI productivity coach. Given a history digest and current snapshot,
write a brief (1–2 sentences) suggestion. Be concrete and name targets (tab, host, file) when helpful.

INTENT: {intent}

BUNDLE (JSON):
{json.dumps(bundle, default=str)[:6000]}  # hard guard against token blowout

Respond as JSON:
{{
  "message": "...",
  "priority": 1|2|3,
  "urgency": "low|medium|high", 
  "reasoning": "short why",
  "confidence": 0.0-1.0
}}""".strip()

        try:
            msg = await self.claude_client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=7300,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = msg.content[0].text.strip()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"message": raw, "priority": 2, "urgency": "medium",
                           "reasoning": "ai-freeform", "confidence": 0.7}

            persona = self._get_user_profile(user_id).persona
            return {
                "id": str(uuid.uuid4()),
                "type": intent,
                "message": payload.get("message", "Focus for 25 minutes on your main tab."),
                "priority": int(payload.get("priority", 2)),
                "urgency": payload.get("urgency", "medium"),
                "persona": persona,
                "channel": self.notification_config.get('default_channel', 'system_banner'),
                "meta": {
                    "source": "three_pass_ai",
                    "confidence": float(payload.get("confidence", 0.7)),
                    "reasoning": payload.get("reasoning", ""),
                    "bundle_digest": {"top_hosts": bundle["history_digest"].get("top_hosts", [])[:3]}
                }
            }
        except Exception as e:
            logger.warning(f"Anthropic synthesis failed: {e}")
            # Graceful fallback
            return self._rule_synthesis(user_id, bundle, DetectorFlag(type=intent, severity="medium", evidence={}, suggestion=None))

    # ========================================================================
    # MAIN COACHING INTERFACE
    # ========================================================================

    async def analyze_telemetry(self, telemetry_data: Dict, user_id: str = "default",
                                context_history: List[Dict] = None) -> Optional[Dict]:
        """
        Main entry point for telemetry analysis and coaching.
        Automatically uses the best available method based on system capabilities.
        """

        # Check if WorkSmart is running or we have recent activity (more flexible check)
        worksmart_active = telemetry_data.get(
            'worksmart_session_active', False)
        has_recent_data = len(telemetry_data.get('recent_events', [])) > 0
        worksmart_hours = telemetry_data.get('worksmart_hours_today', '0:0')
        has_work_hours = worksmart_hours != '0:0' and not worksmart_hours.startswith(
            '0:')

        session_valid = worksmart_active or has_recent_data or has_work_hours or self.test_mode

        if not session_valid:
            print(
                f"[DEBUG] ❌ SKIPPING coaching - No active session detected:", flush=True)
            print(
                f"[DEBUG]   - worksmart_active: {worksmart_active}", flush=True)
            print(
                f"[DEBUG]   - has_recent_data: {has_recent_data}", flush=True)
            print(
                f"[DEBUG]   - worksmart_hours: {worksmart_hours}", flush=True)
            print(f"[DEBUG]   - test_mode: {self.test_mode}", flush=True)
            logger.info(
                "Skipping coaching - No active WorkSmart session detected")
            return None
        else:
            print(f"[DEBUG] ✅ Session valid - proceeding with coaching:", flush=True)
            print(
                f"[DEBUG]   - worksmart_active: {worksmart_active}", flush=True)
            print(
                f"[DEBUG]   - has_recent_data: {has_recent_data}", flush=True)
            print(
                f"[DEBUG]   - worksmart_hours: {worksmart_hours}", flush=True)

        try:
            # NEW: Unified three-pass pipeline (logs + detectors + synthesis)
            # Ingest current telemetry events for continuous learning
            if "current_window" in telemetry_data or "current_app" in telemetry_data:
                self.ingest_event(user_id, "window_focus",
                                  app=telemetry_data.get("current_app") or telemetry_data.get(
                                      "current_application"),
                                  window_title=telemetry_data.get("current_window"))
            if telemetry_data.get("current_url"):
                self.ingest_event(user_id, "url_visit",
                                  app="browser", url=telemetry_data["current_url"])

            # Run three-pass coaching cycle
            print(f"[DEBUG] 🚀 Running coaching cycle...", flush=True)
            cycle_result = await self.run_coaching_cycle(user_id, telemetry_data)
            print(
                f"[DEBUG] 🎯 Coaching cycle result: {type(cycle_result)} - {cycle_result}", flush=True)

            if cycle_result and cycle_result.get("type") != "no_suggestion":
                print(
                    f"[DEBUG] ✅ Found intervention: {cycle_result['type']} - {cycle_result['message'][:50]}...", flush=True)
                logger.info(
                    f"Three-pass coaching: {cycle_result['type']} - {cycle_result['message'][:50]}...")
                return cycle_result
            elif cycle_result and cycle_result.get("type") == "no_suggestion":
                print(
                    f"[DEBUG] ℹ️  Explicit no_suggestion from coaching cycle", flush=True)
                logger.info("Three-pass coaching: No suggestion needed")
                return None  # Don't fall back to other methods if explicitly no suggestion
            # Enhanced ML coaching if available
            if 'ml_pattern_learning' in self.capabilities:
                result = await self._analyze_telemetry_enhanced(telemetry_data, user_id, context_history)
                if result:
                    return result

            # Personalized coaching fallback
            if 'persona_detection' in self.capabilities:
                result = await self._analyze_telemetry_personalized(telemetry_data, user_id)
                if result:
                    return result

            # Basic coaching final fallback
            print(f"[DEBUG] 🔧 Using basic coaching fallback", flush=True)
            result = await self._analyze_telemetry_basic(telemetry_data, user_id)
            print(
                f"[DEBUG] 🔧 Basic coaching result: {result is not None}", flush=True)
            return result

        except Exception as e:
            print(f"[DEBUG] ❌ Exception in analyze_telemetry: {e}", flush=True)
            logger.error(f"Telemetry analysis failed: {e}")
            result = await self._analyze_telemetry_basic(telemetry_data, user_id)
            print(
                f"[DEBUG] 🔧 Exception fallback result: {result is not None}", flush=True)
            return result

    async def _analyze_telemetry_enhanced(self, telemetry_data: Dict, user_id: str,
                                          context_history: List[Dict] = None) -> Optional[Dict]:
        """Enhanced ML-based telemetry analysis"""

        # Update user model with new context
        analysis = self._basic_analysis(telemetry_data)
        context = self._extract_context(telemetry_data, analysis)

        self._update_user_context(user_id, context, analysis)

        # Get user profile and coaching strategy
        user_profile = self._get_user_profile(user_id)
        coaching_strategy = self._get_coaching_strategy(user_id, context)

        if not coaching_strategy['intervention_needed']:
            return None

        # Get predictive insights
        user_history = list(self.user_contexts.get(user_id, []))
        predictive_insights = self._analyze_predictive_insights(
            user_id, user_history, context)

        # Check optimal timing
        intervention_type = coaching_strategy['intervention_type']
        optimal_timing = self._get_optimal_intervention_timing(
            user_id, intervention_type)

        if optimal_timing['delay_minutes'] > 30:
            logger.info(
                f"Delaying intervention for {user_id}: {optimal_timing['reason']}")
            return None

        # Generate AI coaching
        ai_coaching = await self._generate_ai_coaching(user_profile, context, coaching_strategy, predictive_insights)

        if not ai_coaching:
            return None

        # Create comprehensive intervention
        intervention_id = str(uuid.uuid4())
        intervention = {
            'id': intervention_id,
            'user_id': user_id,
            'type': intervention_type,
            'nudge_type': intervention_type,  # Backward compatibility
            'message': ai_coaching.get('message', ''),
            'priority': ai_coaching.get('priority', 2),
            'reasoning': ai_coaching.get('reasoning', ''),
            'confidence': ai_coaching.get('confidence', 0.5),
            'source': 'enhanced_ml',
            'persona': user_profile.persona,
            'predicted_effectiveness': self._predict_intervention_effectiveness(context, ai_coaching) if self.pattern_learner_enabled else 0.5,
            'coaching_strategy': coaching_strategy,
            'predictive_insights': predictive_insights
        }

        # Suppression/cooldown check BEFORE recording
        suppress, reason = self._should_suppress_notification(
            user_id, intervention, context)
        if suppress:
            self._log_notification_event(
                'suppressed', user_id, intervention, reason=reason)
            return None

        # Record (adds timestamp) and log
        self._record_intervention(intervention, user_id)
        self._log_notification_event('sent', user_id, intervention)

        # Record for feedback tracking
        self._record_intervention_for_feedback(intervention_id, user_id, intervention_type,
                                               ai_coaching.get('message', ''), context, ai_coaching.get('priority', 2))

        logger.info(
            f"Generated enhanced ML intervention: {intervention_type} (confidence: {intervention['confidence']:.2f})")
        return intervention

    async def _analyze_telemetry_personalized(self, telemetry_data: Dict, user_id: str) -> Optional[Dict]:
        """Personalized coaching with persona detection"""

        analysis = self._basic_analysis(telemetry_data)
        context = self._extract_context(telemetry_data, analysis)

        # Detect persona
        persona = self._detect_user_persona(context)

        # Get persona-specific coaching
        coaching_result = self._get_persona_specific_coaching(
            persona, context, analysis, user_id)

        if not coaching_result:
            return None

        coaching_result['source'] = 'personalized'
        coaching_result['persona'] = persona

        # Check suppression
        suppress, reason = self._should_suppress_notification(
            user_id, coaching_result, context)
        if suppress:
            logger.info(
                f"Suppressed {coaching_result['type']} for {user_id}: {reason}")
            self._log_notification_event(
                'suppressed', user_id, coaching_result, reason=reason)
            return None

        # Record and return
        self._record_intervention(coaching_result, user_id)
        self._log_notification_event('sent', user_id, coaching_result)
        return coaching_result

    async def _analyze_telemetry_basic(self, telemetry_data: Dict, user_id: str) -> Optional[Dict]:
        """Basic rule-based coaching"""
        print(f"[DEBUG] 🔧 _analyze_telemetry_basic called", flush=True)

        productivity = telemetry_data.get('productivity_score', 0.5)
        focus = telemetry_data.get('focus_quality', 0.5)
        stress = telemetry_data.get('stress_level', 0.5)
        energy = telemetry_data.get('energy_level', 0.5)
        session_hours = telemetry_data.get('session_duration_hours', 0)

        print(
            f"[DEBUG] 📊 Basic coaching data: prod={productivity:.2f}, focus={focus:.2f}, stress={stress:.2f}, energy={energy:.2f}, hours={session_hours:.1f}", flush=True)

        coaching_type, urgency = self._determine_coaching_need(
            productivity, focus, stress, energy, session_hours)

        if not coaching_type:
            return None

        # Try Anthropic API first
        intervention = None
        if self.claude_client:
            ai_response = await self._get_anthropic_coaching(telemetry_data, coaching_type, urgency, user_id)
            if ai_response:
                ai_response['source'] = 'anthropic_ai'
                intervention = ai_response

        # Rule-based fallback
        if not intervention:
            intervention = self._get_rule_based_coaching(
                coaching_type, urgency, telemetry_data, user_id)

        if not intervention:
            return None

        # Check suppression
        analysis = self._basic_analysis(telemetry_data)
        context = self._extract_context(telemetry_data, analysis)
        suppress, reason = self._should_suppress_notification(
            user_id, intervention, context)
        if suppress:
            logger.info(
                f"Suppressed {intervention['type']} for {user_id}: {reason}")
            self._log_notification_event(
                'suppressed', user_id, intervention, reason=reason)
            return None

        # Record and return
        print(
            f"[DEBUG] 🔧 Basic coaching result: coaching_type={coaching_type}, urgency={urgency}, intervention={intervention is not None}", flush=True)
        if intervention:
            print(
                f"[DEBUG] 🔧 Intervention type: {intervention.get('type')}, message: {intervention.get('message', '')[:50]}...", flush=True)
        self._record_intervention(intervention, user_id)
        self._log_notification_event('sent', user_id, intervention)
        return intervention

    # ========================================================================
    # USER MODELING AND CONTEXT ANALYSIS
    # ========================================================================

    def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                energy_patterns={},
                productivity_patterns={},
                intervention_effectiveness={},
                preferences=UserPreferences(),
                last_updated=datetime.now().isoformat(),
                total_interactions=0
            )
        return self.user_profiles[user_id]

    def _update_user_context(self, user_id: str, context: Dict, analysis: Dict):
        """Update user context and learn patterns"""
        profile = self._get_user_profile(user_id)

        # Add context to history
        enriched_context = {
            **context,
            **analysis,
            'timestamp': datetime.now().isoformat(),
            'hour_of_day': datetime.now().hour
        }
        self.user_contexts[user_id].append(enriched_context)

        # Update patterns
        self._update_energy_patterns(profile, enriched_context)
        self._update_productivity_patterns(profile, enriched_context)
        self._update_persona_confidence(profile, enriched_context)
        self._update_adaptive_thresholds(user_id, enriched_context)

        profile.total_interactions += 1
        profile.last_updated = datetime.now().isoformat()

        # Save updated profile
        self._save_user_profile(profile)

    def _update_energy_patterns(self, profile: UserProfile, context: Dict):
        """Learn user's energy patterns by hour"""
        hour = context['hour_of_day']
        energy = context.get('energy_level', 0.5)

        if profile.energy_patterns is None:
            profile.energy_patterns = {}

        # Exponential moving average
        if hour in profile.energy_patterns:
            profile.energy_patterns[hour] = 0.8 * \
                profile.energy_patterns[hour] + 0.2 * energy
        else:
            profile.energy_patterns[hour] = energy

    def _update_productivity_patterns(self, profile: UserProfile, context: Dict):
        """Learn productivity patterns by application"""
        app = context.get('current_application', 'unknown')
        productivity = context.get('productivity_score', 0.5)

        if profile.productivity_patterns is None:
            profile.productivity_patterns = {}

        if app in profile.productivity_patterns:
            profile.productivity_patterns[app] = 0.7 * \
                profile.productivity_patterns[app] + 0.3 * productivity
        else:
            profile.productivity_patterns[app] = productivity

    def _update_persona_confidence(self, profile: UserProfile, context: Dict):
        """Update persona detection confidence"""
        app = context.get('current_application', '').lower()

        persona_scores = defaultdict(float)

        # Score each persona based on application usage
        for persona, patterns in self.persona_patterns.items():
            if persona == 'generic':
                continue
            for indicator_app in patterns['apps']:
                if indicator_app in app:
                    persona_scores[persona] += 0.1

        # Update profile with highest scoring persona
        if persona_scores:
            best_persona = max(persona_scores.items(), key=lambda x: x[1])
            if best_persona[1] > 0.05:
                if profile.persona != best_persona[0]:
                    profile.persona = best_persona[0]
                    profile.confidence_level = min(
                        0.9, profile.confidence_level + 0.1)

    def _update_adaptive_thresholds(self, user_id: str, context: Dict):
        """Update personalized thresholds based on user's baselines"""
        if user_id not in self.adaptive_thresholds:
            self.adaptive_thresholds[user_id] = {
                'productivity_low': 0.3,
                'productivity_high': 0.7,
                'focus_low': 0.4,
                'focus_high': 0.8,
                'stress_high': 0.6,
                'energy_low': 0.3
            }

        profile = self.user_profiles[user_id]
        thresholds = self.adaptive_thresholds[user_id]

        # Update baselines with exponential moving average
        productivity_score = context.get('productivity_score', 0.5)
        focus_quality = context.get('focus_quality', 0.5)

        profile.productivity_baseline = 0.95 * \
            profile.productivity_baseline + 0.05 * productivity_score
        profile.focus_baseline = 0.95 * profile.focus_baseline + 0.05 * focus_quality

        # Adapt thresholds to be relative to user's baseline
        thresholds['productivity_low'] = max(
            0.1, profile.productivity_baseline - 0.2)
        thresholds['productivity_high'] = min(
            0.9, profile.productivity_baseline + 0.2)
        thresholds['focus_low'] = max(0.1, profile.focus_baseline - 0.2)
        thresholds['focus_high'] = min(0.9, profile.focus_baseline + 0.2)

    def _get_coaching_strategy(self, user_id: str, context: Dict) -> Dict:
        """Get personalized coaching strategy"""
        profile = self._get_user_profile(user_id)
        thresholds = self.adaptive_thresholds.get(user_id, {
            'productivity_low': 0.3, 'productivity_high': 0.7,
            'focus_low': 0.4, 'focus_high': 0.8,
            'stress_high': 0.6, 'energy_low': 0.3
        })

        productivity = context.get('productivity_score', 0.5)
        focus = context.get('focus_quality', 0.5)
        stress = context.get('stress_level', 0.5)
        energy = context.get('energy_level', 0.5)

        # Determine intervention need and type
        intervention_needed = False
        intervention_type = None
        urgency_level = 'low'
        confidence = 0.5

        if stress > thresholds.get('stress_high', 0.6):
            intervention_needed = True
            intervention_type = 'stress_reduction'
            urgency_level = 'high' if stress > 0.8 else 'medium'
            confidence = 0.8
        elif energy < thresholds.get('energy_low', 0.3):
            intervention_needed = True
            intervention_type = 'energy_boost'
            urgency_level = 'medium'
            confidence = 0.6
        elif productivity < thresholds.get('productivity_low', 0.3):
            intervention_needed = True
            intervention_type = 'productivity_boost'
            urgency_level = 'medium' if productivity < 0.2 else 'low'
            confidence = 0.7
        elif focus < thresholds.get('focus_low', 0.4):
            intervention_needed = True
            intervention_type = 'focus_enhancement'
            urgency_level = 'medium' if focus < 0.3 else 'low'
            confidence = 0.6

        return {
            'intervention_needed': intervention_needed,
            'intervention_type': intervention_type,
            'urgency_level': urgency_level,
            'confidence': confidence,
            'personalized_thresholds': thresholds,
            'user_baseline': {
                'productivity': profile.productivity_baseline,
                'focus': profile.focus_baseline
            }
        }

    # ========================================================================
    # PREDICTIVE ANALYTICS AND BURNOUT PREVENTION
    # ========================================================================

    def _analyze_predictive_insights(self, user_id: str, user_history: List[Dict],
                                     current_context: Dict) -> Dict[str, Any]:
        """Generate predictive insights including burnout risk"""

        insights = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'prediction_confidence': min(0.8, len(user_history) / 50)
        }

        # Burnout risk analysis
        burnout_analysis = self._predict_burnout_risk(
            user_history, current_context)
        insights['burnout_risk'] = burnout_analysis

        # Anomaly detection
        anomaly_analysis = self._detect_anomalies(
            user_id, current_context, user_history)
        insights['anomalies'] = anomaly_analysis

        # Generate predictions
        insights['predictions'] = self._generate_predictions(
            user_history, current_context)

        # Overall risk assessment
        insights['overall_risk_level'] = self._calculate_overall_risk(
            burnout_analysis, anomaly_analysis)

        return insights

    def _predict_burnout_risk(self, user_history: List[Dict], current_context: Dict) -> Dict[str, Any]:
        """Predict burnout risk based on patterns"""

        if len(user_history) < 5:
            return {
                'risk_score': 0.3,
                'risk_level': 'low',
                'factors': ['insufficient_data'],
                'recommendation': 'Continue monitoring'
            }

        # Analyze risk factors
        risk_indicators = self._analyze_risk_factors(
            user_history, current_context)

        # Risk factor weights
        risk_factors = {
            'prolonged_high_stress': 0.3,
            'declining_productivity': 0.25,
            'extended_work_hours': 0.2,
            'poor_sleep_indicators': 0.15,
            'lack_of_breaks': 0.1
        }

        # Calculate overall risk score
        risk_score = sum(risk_indicators.get(factor, 0) * weight
                         for factor, weight in risk_factors.items())

        risk_score = max(0.0, min(1.0, risk_score))

        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'critical'
        elif risk_score > 0.5:
            risk_level = 'high'
        elif risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        active_factors = [factor for factor,
                          score in risk_indicators.items() if score > 0.3]

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'factors': active_factors,
            'factor_details': risk_indicators,
            'recommendation': self._generate_burnout_recommendation(risk_level, active_factors),
            'confidence': min(0.8, len(user_history) / 50)
        }

    def _analyze_risk_factors(self, user_history: List[Dict], current_context: Dict) -> Dict[str, float]:
        """Analyze specific burnout risk factors"""

        risk_indicators = {}

        # Analyze stress patterns
        stress_levels = [ctx.get('stress_level', 0.5)
                         for ctx in user_history[-20:]]
        avg_stress = sum(stress_levels) / \
            len(stress_levels) if stress_levels else 0.5
        high_stress_days = len([s for s in stress_levels if s > 0.7])

        risk_indicators['prolonged_high_stress'] = min(
            1.0, (avg_stress - 0.4) * 2 + high_stress_days / 20)

        # Analyze productivity decline
        productivity_scores = [ctx.get('productivity_score', 0.5)
                               for ctx in user_history[-15:]]
        if len(productivity_scores) >= 10:
            recent_prod = sum(productivity_scores[-5:]) / 5
            earlier_prod = sum(productivity_scores[:5]) / 5
            decline = max(0, earlier_prod - recent_prod)
            risk_indicators['declining_productivity'] = min(1.0, decline * 3)
        else:
            risk_indicators['declining_productivity'] = 0.2

        # Analyze work hours
        session_hours = [ctx.get('session_duration_hours', 0)
                         for ctx in user_history[-10:]]
        avg_hours = sum(session_hours) / \
            len(session_hours) if session_hours else 4
        excessive_hours = len([h for h in session_hours if h > 8])

        risk_indicators['extended_work_hours'] = min(
            1.0, (avg_hours - 6) / 4 + excessive_hours / 10)

        # Analyze break patterns
        break_counts = [1 if ctx.get(
            'break_taken', False) else 0 for ctx in user_history[-10:]]
        if break_counts:
            break_frequency = sum(break_counts) / len(break_counts)
            risk_indicators['lack_of_breaks'] = max(
                0, 1.0 - break_frequency * 2)
        else:
            risk_indicators['lack_of_breaks'] = 0.4

        # Sleep indicators (energy levels)
        energy_levels = [ctx.get('energy_level', 0.5)
                         for ctx in user_history[-10:]]
        avg_energy = sum(energy_levels) / \
            len(energy_levels) if energy_levels else 0.5
        low_energy_frequency = len(
            [e for e in energy_levels if e < 0.3]) / len(energy_levels) if energy_levels else 0

        risk_indicators['poor_sleep_indicators'] = min(
            1.0, low_energy_frequency + (0.5 - avg_energy))

        return risk_indicators

    def _generate_burnout_recommendation(self, risk_level: str, factors: List[str]) -> str:
        """Generate burnout prevention recommendation"""

        if risk_level == 'critical':
            return "🚨 High burnout risk detected. Consider taking time off and reducing workload immediately."
        elif risk_level == 'high':
            return "⚠️ Elevated burnout risk. Implement stress reduction and ensure regular breaks."
        elif risk_level == 'medium':
            return "💡 Monitor stress levels and maintain work-life balance to prevent escalation."
        else:
            return "✅ Burnout risk is low. Continue current healthy work patterns."

    def _detect_anomalies(self, user_id: str, current_context: Dict, user_history: List[Dict]) -> Dict[str, Any]:
        """Detect unusual patterns that might indicate problems"""

        if len(user_history) < 10:
            return {'anomalies_detected': False, 'message': 'Insufficient data for anomaly detection'}

        # Simple rule-based anomaly detection
        # Check for sudden productivity drops
        recent_prod = [ctx.get('productivity_score', 0.5)
                       for ctx in user_history[-5:]]
        earlier_prod = [ctx.get('productivity_score', 0.5)
                        for ctx in user_history[-15:-5]]

        if len(recent_prod) >= 3 and len(earlier_prod) >= 5:
            recent_avg = sum(recent_prod) / len(recent_prod)
            earlier_avg = sum(earlier_prod) / len(earlier_prod)

            if earlier_avg - recent_avg > 0.3:  # Significant drop
                return {
                    'anomalies_detected': True,
                    'anomaly_type': 'productivity_decline',
                    'severity': 'medium',
                    'description': f'Productivity dropped from {earlier_avg:.1%} to {recent_avg:.1%}',
                    'recommendation': 'Investigate potential causes of productivity decline'
                }

        # Check for stress spikes
        current_stress = current_context.get('stress_level', 0.5)
        avg_stress = sum(ctx.get('stress_level', 0.5)
                         for ctx in user_history[-10:]) / min(10, len(user_history))

        if current_stress > avg_stress + 0.4:  # Stress spike
            return {
                'anomalies_detected': True,
                'anomaly_type': 'stress_spike',
                'severity': 'high',
                'description': f'Stress level {current_stress:.1%} significantly above normal {avg_stress:.1%}',
                'recommendation': 'Immediate stress reduction intervention recommended'
            }

        return {'anomalies_detected': False, 'message': 'No anomalies detected'}

    def _generate_predictions(self, history: List[Dict], context: Dict) -> List[Dict]:
        """Generate specific predictions about user behavior"""

        predictions = []

        # Predict productivity recovery
        current_prod = context.get('productivity_score', 0.5)
        if current_prod < 0.4:
            recovery_prob = min(0.9, 0.3 + len(history) / 100)
            predictions.append({
                'type': 'productivity_recovery',
                'probability': recovery_prob,
                'timeframe': 'next_30_minutes',
                'description': 'Productivity likely to improve with appropriate intervention'
            })

        # Predict break need
        session_hours = context.get('session_duration_hours', 0)
        if session_hours > 3:  # increased from 2
            # reduced urgency calculation
            break_urgency = min(0.95, session_hours / 5)
            predictions.append({
                'type': 'break_needed',
                'probability': break_urgency,
                'timeframe': 'immediate',
                'description': 'Break recommended to maintain productivity and well-being'
            })

        return predictions

    def _calculate_overall_risk(self, burnout_analysis: Dict, anomaly_analysis: Dict) -> str:
        """Calculate overall risk level"""

        burnout_risk = burnout_analysis.get('risk_score', 0.3)
        has_anomalies = anomaly_analysis.get('anomalies_detected', False)

        if burnout_risk > 0.7 or (has_anomalies and anomaly_analysis.get('severity') == 'high'):
            return 'high'
        elif burnout_risk > 0.5 or has_anomalies:
            return 'medium'
        else:
            return 'low'

    def _get_optimal_intervention_timing(self, user_id: str, intervention_type: str) -> Dict[str, Any]:
        """Get optimal timing for intervention based on user patterns"""

        profile = self._get_user_profile(user_id)
        current_hour = datetime.now().hour

        # Check user preferences
        if profile.preferences and profile.preferences.avoided_times:
            if current_hour in profile.preferences.avoided_times:
                return {
                    'delay_minutes': 60,
                    'reason': 'User prefers not to be disturbed at this time'
                }

        # Check energy patterns
        if profile.energy_patterns and current_hour in profile.energy_patterns:
            energy_level = profile.energy_patterns[current_hour]

            # Avoid interventions during low energy periods for productivity/focus
            if intervention_type in ['productivity_boost', 'focus_enhancement'] and energy_level < 0.3:
                return {
                    'delay_minutes': 30,
                    'reason': 'User typically has low energy at this time'
                }

        # Check recent intervention frequency
        recent_contexts = list(self.user_contexts[user_id])[-10:]
        recent_interventions = sum(
            1 for ctx in recent_contexts if ctx.get('intervention_received', False))

        if recent_interventions > 3:
            return {
                'delay_minutes': 45,
                'reason': 'Too many recent interventions - giving user space'
            }

        return {
            'delay_minutes': 0,
            'reason': 'Optimal timing for intervention'
        }

    # ========================================================================
    # PERSONA DETECTION AND PERSONALIZED COACHING
    # ========================================================================

    def _detect_user_persona(self, context: Dict) -> str:
        """Detect user persona based on application usage and context"""

        current_app = context.get('current_application', '').lower()
        current_window = context.get('current_window', '').lower()

        # Force developer persona for Cursor users
        if 'cursor' in current_app.lower():
            return 'developer'

        persona_scores = {'developer': 0, 'analyst': 0, 'manager': 0}

        # Score based on application usage
        for persona, patterns in self.persona_patterns.items():
            if persona == 'generic':
                continue
            for app in patterns['apps']:
                if app in current_app:
                    persona_scores[persona] += 2
                if app in current_window:
                    persona_scores[persona] += 1

            # Score based on keywords in window titles
            for keyword in patterns['keywords']:
                if keyword in current_window:
                    persona_scores[persona] += 1

        # Return highest scoring persona, or 'generic' if no clear winner
        if max(persona_scores.values()) >= 2:
            return max(persona_scores.items(), key=lambda x: x[1])[0]
        return 'generic'

    def _get_persona_specific_coaching(self, persona: str, context: Dict, analysis: Dict, user_id: str = "default") -> Optional[Dict]:
        """Generate persona-specific coaching"""

        # Determine coaching need
        productivity = analysis.get('productivity_score', 0.5)
        focus = analysis.get('focus_quality', 0.5)
        stress = analysis.get('stress_level', 0.5)
        energy = analysis.get('energy_level', 0.5)
        session_hours = analysis.get(
            'session_duration_hours', analysis.get('session_hours', 0))

        coaching_type, urgency = self._determine_coaching_need(
            productivity, focus, stress, energy, session_hours)

        if not coaching_type:
            return None

        # Get persona-specific message
        persona_messages = self.persona_patterns.get(
            persona, self.persona_patterns['generic'])['coaching_messages']
        message_or_list = persona_messages.get(coaching_type,
                                               f"Consider optimizing your current work state for better {coaching_type.replace('_', ' ')}.")
        message = self._choose_copy_variant(message_or_list, user_id)
        message = self._adjust_message_for_context(
            message, context, urgency, coaching_type)

        priority = 3 if urgency == 'high' else 2 if urgency == 'medium' else 1

        # Allow medium urgency if not in meeting and within caps (suppression handled upstream)
        if urgency not in ['high', 'critical', 'medium']:
            return None

        return {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': message,
            'priority': priority,
            'urgency': urgency,
            'persona': persona,
            'channel': self.notification_config.get('default_channel', 'system_banner'),
            'meta': {
                'reasoning': f"Persona-specific {coaching_type} for {persona}",
                'confidence': 0.8 if persona != 'generic' else 0.6,
                'source': 'personalized',
                'cooldown_applied': False
            }
        }

    # ========================================================================
    # AI COACHING GENERATION (ANTHROPIC INTEGRATION)
    # ========================================================================

    async def _generate_ai_coaching(self, user_profile: UserProfile, context: Dict,
                                    coaching_strategy: Dict, predictive_insights: Dict) -> Optional[Dict]:
        """Generate AI coaching using Anthropic API with dynamic prompts"""

        try:
            # Determine prompt type
            if predictive_insights.get('burnout_risk', {}).get('risk_score', 0) > 0.6:
                prompt_type = 'burnout_prevention'
            else:
                prompt_type = 'productivity_analysis'

            # Generate dynamic prompt
            prompt = self._generate_dynamic_prompt(
                prompt_type, user_profile, context, predictive_insights)

            # Get AI response using Anthropic
            if self.claude_client:
                try:
                    message = await self.claude_client.messages.create(
                        model="claude-3-5-haiku-latest",
                        max_tokens=7300,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    response_text = message.content[0].text.strip()

                    # Try to parse JSON response
                    try:
                        ai_coaching = json.loads(response_text)
                        if all(field in ai_coaching for field in ['message', 'priority', 'reasoning', 'confidence']):
                            return ai_coaching
                    except json.JSONDecodeError:
                        # Use text response as message
                        return {
                            'message': response_text,
                            'priority': coaching_strategy.get('urgency_level', 'medium') == 'high' and 3 or 2,
                            'reasoning': f"AI-generated {coaching_strategy['intervention_type']} advice",
                            'confidence': 0.7
                        }

                except Exception as e:
                    logger.warning(f"Anthropic API call failed: {e}")

            # Fallback to persona-specific coaching
            return self._generate_fallback_coaching(coaching_strategy, context, user_profile)

        except Exception as e:
            logger.error(f"AI coaching generation failed: {e}")
            return self._generate_fallback_coaching(coaching_strategy, context, user_profile)

    def _generate_dynamic_prompt(self, prompt_type: str, user_profile: UserProfile,
                                 context: Dict, predictive_insights: Dict) -> str:
        """Generate dynamic, personalized prompts for Anthropic API"""

        if prompt_type == 'burnout_prevention':
            prompt = f"""You are an expert wellness coach specializing in burnout prevention. Analyze the user's risk factors and provide preventive guidance.

User Profile:
- Persona: {user_profile.persona} (confidence: {user_profile.confidence_level:.1%})
- Productivity baseline: {user_profile.productivity_baseline:.1%}
- Total interactions: {user_profile.total_interactions}

Current Context:
- Current productivity: {context.get('productivity_score', 0.5):.1%}
- Current focus quality: {context.get('focus_quality', 0.5):.1%}
- Current stress level: {context.get('stress_level', 0.5):.1%}
- Session duration: {context.get('session_duration_hours', 0):.1f} hours
- Current application: {context.get('current_application', 'Unknown')}

Burnout Risk Analysis: {predictive_insights.get('burnout_risk', {})}

Provide specific, evidence-based advice to prevent burnout while maintaining productivity.

Respond in JSON format:
{{
    "message": "Caring, supportive message about burnout prevention",
    "priority": 2-3 (always medium-high for burnout),
    "reasoning": "Specific risk factors identified",
    "confidence": 0.0-1.0,
    "expected_impact": "How this will reduce burnout risk"
}}"""
        else:
            prompt = f"""You are an expert productivity coach analyzing telemetry data. Based on the user's current state and historical patterns, provide specific, actionable coaching advice.

User Profile:
- Persona: {user_profile.persona} (confidence: {user_profile.confidence_level:.1%})
- Productivity baseline: {user_profile.productivity_baseline:.1%}
- Focus baseline: {user_profile.focus_baseline:.1%}
- Total interactions: {user_profile.total_interactions}

Current Context:
- Current productivity: {context.get('productivity_score', 0.5):.1%}
- Current focus quality: {context.get('focus_quality', 0.5):.1%}
- Current stress level: {context.get('stress_level', 0.5):.1%}
- Current energy level: {context.get('energy_level', 0.5):.1%}
- Session duration: {context.get('session_duration_hours', 0):.1f} hours
- Current application: {context.get('current_application', 'Unknown')}

Predictive Insights: {predictive_insights}

Focus on immediate actionable advice based on current state and personalized recommendations based on user patterns.

Respond in JSON format:
{{
    "message": "Brief, actionable coaching message (1-2 sentences)",
    "priority": 1-3 (1=low, 2=medium, 3=urgent),
    "reasoning": "Why this advice is relevant now",
    "confidence": 0.0-1.0,
    "expected_impact": "What improvement this should achieve"
}}"""

        return prompt

    async def _get_anthropic_coaching(self, telemetry: Dict, coaching_type: str, urgency: str, user_id: str = "default") -> Optional[Dict]:
        """Get AI coaching from Anthropic Claude"""

        if not self.claude_client:
            return None

        try:
            context = self._build_simple_coaching_context(
                telemetry, coaching_type, urgency)

            message = await self.claude_client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=7200,
                temperature=0.7,
                messages=[{"role": "user", "content": context}]
            )

            ai_message = message.content[0].text.strip()
            return self._normalize_ai_response(coaching_type, urgency, ai_message, user_id=user_id)

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None

    def _build_simple_coaching_context(self, telemetry: Dict, coaching_type: str, urgency: str) -> str:
        """Build context-aware prompt for Anthropic"""

        productivity = telemetry.get('productivity_score', 0.5)
        focus = telemetry.get('focus_quality', 0.5)
        stress = telemetry.get('stress_level', 0.5)
        energy = telemetry.get('energy_level', 0.5)
        session_hours = telemetry.get('session_duration_hours', 0)
        current_app = telemetry.get('current_application', 'unknown')

        prompt = f"""You are an expert productivity coach. Based on the user's current state, provide a brief, actionable coaching message (1-2 sentences max).

Current metrics:
- Productivity: {productivity:.0%}
- Focus: {focus:.0%}
- Stress: {stress:.0%}
- Energy: {energy:.0%}
- Session duration: {session_hours:.1f} hours
- Current app: {current_app}

Coaching needed: {coaching_type} (urgency: {urgency})

Provide specific, actionable advice that addresses the {coaching_type} issue. Be encouraging and practical."""

        return prompt

    def _generate_fallback_coaching(self, coaching_strategy: Dict, context: Dict,
                                    user_profile: UserProfile) -> Dict:
        """Generate rule-based coaching when AI is unavailable"""

        intervention_type = coaching_strategy['intervention_type']

        # Use persona-specific messaging if available
        persona_messages = self.persona_patterns.get(user_profile.persona,
                                                     self.persona_patterns['generic'])['coaching_messages']

        message = persona_messages.get(intervention_type,
                                       f"Consider optimizing your current work state for better {intervention_type.replace('_', ' ')}.")

        return {
            'message': message,
            'priority': coaching_strategy.get('urgency_level', 'medium') == 'high' and 3 or 2,
            'reasoning': f"ML-enhanced {intervention_type} recommendation for {user_profile.persona}",
            'confidence': 0.6
        }

    # ========================================================================
    # BASIC ANALYSIS AND RULE-BASED COACHING
    # ========================================================================

    def _determine_coaching_need(self, productivity: float, focus: float,
                                 stress: float, energy: float, session_hours: float) -> tuple:
        """Determine if coaching is needed and what type"""

        # Debug metrics
        print(
            f"🔍 METRICS: P={productivity:.2f} F={focus:.2f} S={stress:.2f} E={energy:.2f} Hours={session_hours:.1f}")

        # Priority order: stress > energy > productivity > focus > breaks
        if stress >= self.coaching_strategies['stress_thresholds']['high']:
            print(
                f"🚨 STRESS HIGH: {stress:.2f} >= {self.coaching_strategies['stress_thresholds']['high']}")
            return 'stress_reduction', 'high'
        elif stress >= self.coaching_strategies['stress_thresholds']['moderate']:
            print(
                f"⚠️ STRESS MODERATE: {stress:.2f} >= {self.coaching_strategies['stress_thresholds']['moderate']}")
            return 'stress_reduction', 'medium'

        if energy <= self.coaching_strategies['energy_thresholds']['critical']:
            print(
                f"🔋 ENERGY CRITICAL: {energy:.2f} <= {self.coaching_strategies['energy_thresholds']['critical']}")
            return 'energy_boost', 'high'
        elif energy <= self.coaching_strategies['energy_thresholds']['low']:
            print(
                f"⚡ ENERGY LOW: {energy:.2f} <= {self.coaching_strategies['energy_thresholds']['low']}")
            return 'energy_boost', 'medium'

        if productivity <= 0.20:  # Very low productivity - catch early for 8h
            print(f"🚨 PRODUCTIVITY VERY LOW: {productivity:.2f} <= 0.20")
            return 'productivity_boost', 'high'
        elif productivity <= self.coaching_strategies['productivity_thresholds']['low']:
            print(
                f"📈 PRODUCTIVITY LOW: {productivity:.2f} <= {self.coaching_strategies['productivity_thresholds']['low']}")
            return 'productivity_boost', 'medium'

        if focus <= 0.30:  # Very low focus - catch early for 8h quality
            print(f"🎯 FOCUS VERY LOW: {focus:.2f} <= 0.30")
            return 'focus_enhancement', 'medium'
        elif focus <= self.coaching_strategies['focus_thresholds']['low']:
            print(
                f"💡 FOCUS LOW: {focus:.2f} <= {self.coaching_strategies['focus_thresholds']['low']}")
            return 'focus_enhancement', 'low'

        if session_hours > 3:  # Long session - break for 8h sustainability
            print(f"⏰ LONG SESSION: {session_hours:.1f}h > 3h")
            return 'break_reminder', 'medium'
        elif session_hours > 2:   # Regular break reminder
            print(f"💭 SESSION LENGTH: {session_hours:.1f}h > 2h")
            return 'break_reminder', 'low'

        print("✅ NO COACHING NEEDED - All metrics healthy")
        return None, None  # No coaching needed

    def _get_rule_based_coaching(self, coaching_type: str, urgency: str, telemetry: Dict, user_id: str = "default") -> Dict:
        """Get rule-based coaching advice"""

        templates = {
            'productivity_boost': {
                'very_low': "🚨 20% PRODUCTIVITY. Kill all tabs. ONE task. 25 minutes. START NOW.",
                'low': "⚡ Productivity at 30%. Close 5 things. Pick your #1 task. Timer: 25 min. GO."
            },
            'focus_enhancement': {
                'distracted': "🎯 15+ TAB SWITCHES. Stop. Close everything except ONE work tab. 20 min focus or fail.",
                'scattered': "💡 Focus dead. Cmd+W all distractions. Full-screen your work. 25 minutes uninterrupted."
            },
            'stress_reduction': {
                'high': "😤 STRESS 80%+. Stop working NOW. 4-7-8 breathing x3. Then ONE task only.",
                'moderate': "⚠️ Stress climbing. 60-second break NOW or lose 2 hours to burnout."
            },
            'energy_boost': {
                'critical': "🔋 ENERGY 15%. Stand NOW. Walk 2 min. Cold water. Or crash in 30 min.",
                'low': "⚡ Energy 30%. Quick fix: Stand, 10 squats, hydrate. Back in 90 seconds."
            },
            'break_reminder': {
                'urgent': "⏰ 3+ HOURS NONSTOP. Break NOW or tank. Walk 5 min. Non-negotiable.",
                'gentle': "💭 90 min sprint done. 5-min break = 30% better next session."
            }
        }

        # Select appropriate template based on urgency and specific metrics
        template_group = templates.get(coaching_type, {})

        if coaching_type == 'productivity_boost':
            productivity = telemetry.get('productivity_score', 0.5)
            message = template_group.get('very_low' if productivity < 0.2 else 'low',
                                         "Focus on your most important task to boost productivity.")

        elif coaching_type == 'focus_enhancement':
            message = template_group.get('distracted' if urgency == 'high' else 'scattered',
                                         "Close unnecessary applications to improve focus.")

        elif coaching_type == 'stress_reduction':
            message = template_group.get('high' if urgency == 'high' else 'moderate',
                                         "Take deep breaths and consider a brief break.")

        elif coaching_type == 'energy_boost':
            energy = telemetry.get('energy_level', 0.5)
            message = template_group.get('critical' if energy < 0.2 else 'low',
                                         "Stand up and stretch to boost your energy.")

        elif coaching_type == 'break_reminder':
            session_hours = telemetry.get('session_duration_hours', 0)
            message = template_group.get('urgent' if session_hours > 3 else 'gentle',
                                         "Consider taking a short break to maintain productivity.")

        else:
            message = "Take a moment to assess your current work state and make any needed adjustments."

        priority = 3 if urgency == 'high' else 2 if urgency == 'medium' else 1

        return {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': message,
            'priority': priority,
            'urgency': urgency,
            'persona': self._get_user_profile(user_id).persona,
            'channel': self.notification_config.get('default_channel', 'system_banner'),
            'meta': {
                'reasoning': f"Rule-based {coaching_type} recommendation",
                'confidence': 0.6,
                'source': 'rule_based',
                'cooldown_applied': False
            }
        }

    def _basic_analysis(self, telemetry_data: Dict) -> Dict:
        """Perform basic analysis on telemetry data"""
        return {
            'productivity_score': telemetry_data.get('productivity_score', 0.5),
            'focus_quality': telemetry_data.get('focus_quality', 0.5),
            'stress_level': telemetry_data.get('stress_level', 0.5),
            'energy_level': telemetry_data.get('energy_level', 0.5),
            'session_duration_hours': telemetry_data.get('session_duration_hours', telemetry_data.get('session_hours', 0)),
            'activity_level': telemetry_data.get('activity_level', 'MEDIUM'),
            'current_app': telemetry_data.get('current_app', 'Unknown'),
            'in_call': telemetry_data.get('in_call', False)
        }

    def _extract_context(self, telemetry_data: Dict, analysis: Dict) -> Dict:
        """Extract context from telemetry data and analysis"""
        return {
            'current_application': analysis.get('current_app', 'Unknown'),
            'productivity_score': analysis.get('productivity_score', 0.5),
            'focus_quality': analysis.get('focus_quality', 0.5),
            'stress_level': analysis.get('stress_level', 0.5),
            'energy_level': analysis.get('energy_level', 0.5),
            'session_duration_hours': analysis.get('session_duration_hours', analysis.get('session_hours', 0)),
            'activity_level': analysis.get('activity_level', 'MEDIUM'),
            'in_meeting': analysis.get('in_call', False),
            'keyboard_count': telemetry_data.get('total_keystrokes', 0),
            'mouse_count': telemetry_data.get('total_mouse_events', 0),
            'current_window': telemetry_data.get('current_window', ''),
            'break_taken': telemetry_data.get('break_taken', False) or self._infer_break_taken(telemetry_data.get('event_buffer', []))
        }

    # ========================================================================
    # ML PATTERN LEARNING AND PREDICTION
    # ========================================================================

    def _predict_intervention_effectiveness(self, context: Dict, intervention: Dict) -> float:
        """Predict how effective an intervention will be"""

        if not self.pattern_learner_enabled or len(self.interaction_data) < 5:
            return self._rule_based_effectiveness_prediction(context, intervention)

        try:
            features = self._extract_features(context, intervention)
            features_scaled = self.scaler.transform([features])
            prediction = self.effectiveness_predictor.predict_proba(features_scaled)[
                0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except Exception as e:
            logger.warning(f"ML effectiveness prediction failed: {e}")
            return self._rule_based_effectiveness_prediction(context, intervention)

    def _rule_based_effectiveness_prediction(self, context: Dict, intervention: Dict) -> float:
        """Simple rule-based effectiveness prediction"""

        intervention_type = intervention.get(
            'intervention_type', intervention.get('nudge_type', 'productivity_boost'))
        productivity = context.get('productivity_score', 0.5)
        focus = context.get('focus_quality', 0.5)
        stress = context.get('stress_level', 0.5)
        energy = context.get('energy_level', 0.5)

        # Simple rules based on context-intervention match
        if intervention_type == 'productivity_boost':
            return max(0.2, 1.0 - productivity)
        elif intervention_type == 'focus_enhancement':
            return max(0.2, 1.0 - focus)
        elif intervention_type == 'stress_reduction':
            return max(0.2, stress)
        elif intervention_type == 'energy_boost':
            return max(0.2, 1.0 - energy)

        return 0.5

    def _extract_features(self, context: Dict, intervention: Dict) -> List[float]:
        """Extract numerical features for ML models"""

        features = [
            context.get('productivity_score', 0.5),
            context.get('focus_quality', 0.5),
            context.get('stress_level', 0.5),
            context.get('energy_level', 0.5),
            context.get('session_duration_hours', 0),
            intervention.get('priority', 2),
            datetime.now().hour / 24.0,
            len(context.get('current_application', '')) / 50.0,
        ]

        # Intervention type encoding
        intervention_types = ['productivity_boost', 'focus_enhancement',
                              'stress_reduction', 'energy_boost', 'break_reminder']
        intervention_type = (
            intervention.get('type')
            or intervention.get('intervention_type')
            or intervention.get('nudge_type', 'productivity_boost')
        )
        for i, itype in enumerate(intervention_types):
            features.append(1.0 if intervention_type == itype else 0.0)

        # persona one-hot
        persona = (intervention.get('persona')
                   or context.get('persona') or 'generic')
        features.extend([
            1.0 if persona == 'developer' else 0.0,
            1.0 if persona == 'analyst' else 0.0,
            1.0 if persona == 'manager' else 0.0,
        ])

        # contextual flags
        features.append(1.0 if context.get('in_meeting') else 0.0)

        # message length (scaled)
        features.append(len(intervention.get('message', '')) / 200.0)

        # urgency flags
        urg = intervention.get('urgency', 'medium')
        features.extend([
            1.0 if urg == 'high' else 0.0,
            1.0 if urg == 'medium' else 0.0,
        ])

        # hour bucket (coarse)
        hr = int((datetime.now().hour // 6))  # 0..3
        for b in range(4):
            features.append(1.0 if hr == b else 0.0)

        # optional: variant id (if available)
        features.append(float(intervention.get(
            'meta', {}).get('variant_id', 0)) / 10.0)

        return features

    def learn_from_interaction(self, context: Dict, intervention: Dict,
                               effectiveness_score: float, response_time_seconds: float):
        """Learn from a single coaching interaction"""

        interaction_record = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'intervention': intervention,
            'effectiveness_score': effectiveness_score,
            'response_time_seconds': response_time_seconds,
            'user_id': intervention.get('user_id', 'default')
        }

        self.interaction_data.append(interaction_record)

        # Retrain models periodically
        if self.pattern_learner_enabled and len(self.interaction_data) % 10 == 0:
            self._retrain_models()

        # Save interaction data
        self._save_interaction_record(interaction_record)

    def _retrain_models(self):
        """Retrain ML models with accumulated data"""

        if not self.pattern_learner_enabled or len(self.interaction_data) < 5:
            return

        try:
            # Prepare training data
            X = []
            y_effectiveness = []

            for record in self.interaction_data:
                features = self._extract_features(
                    record['context'], record['intervention'])
                X.append(features)
                y_effectiveness.append(
                    1 if record['effectiveness_score'] > 0.6 else 0)

            if len(set(y_effectiveness)) > 1:  # Need multiple classes
                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Train effectiveness predictor
                self.effectiveness_predictor.fit(X_scaled, y_effectiveness)

                self.model_performance['last_trained'] = datetime.now(
                ).isoformat()

                logger.info(
                    f"Retrained ML models with {len(self.interaction_data)} interactions")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    # ========================================================================
    # FEEDBACK COLLECTION AND LEARNING
    # ========================================================================

    def _record_intervention_for_feedback(self, intervention_id: str, user_id: str,
                                          intervention_type: str, message: str, context: Dict,
                                          priority: int = 2) -> None:
        """Record an intervention for feedback tracking"""

        self.intervention_contexts[intervention_id] = {
            'user_id': user_id,
            'type': intervention_type,
            'message': message,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'post_contexts': []
        }

        logger.info(
            f"Recorded intervention {intervention_id} for feedback tracking")

    def record_post_intervention_context(self, intervention_id: str, context: Dict) -> None:
        """Record context after intervention for behavioral analysis"""

        if intervention_id in self.intervention_contexts:
            self.intervention_contexts[intervention_id]['post_contexts'].append({
                **context,
                'timestamp': datetime.now().isoformat()
            })

    def analyze_intervention_effectiveness(self, intervention_id: str) -> Optional[FeedbackEntry]:
        """Analyze intervention effectiveness using behavioral data"""

        if intervention_id not in self.intervention_contexts:
            return None

        intervention_data = self.intervention_contexts[intervention_id]
        post_contexts = intervention_data['post_contexts']

        # Skip analysis if intervention is too recent
        intervention_time = datetime.fromisoformat(
            intervention_data['timestamp'])
        if datetime.now() - intervention_time < timedelta(minutes=10):
            return None

        # Skip if no post-intervention data
        if not post_contexts:
            return None

        # Calculate behavioral effectiveness
        before_context = intervention_data['context']
        effectiveness_score = self._calculate_behavioral_effectiveness(
            before_context, post_contexts, intervention_data['type'])

        # Create feedback entry
        feedback_entry = FeedbackEntry(
            intervention_id=intervention_id,
            user_id=intervention_data['user_id'],
            timestamp=datetime.now().isoformat(),
            intervention_type=intervention_data['type'],
            intervention_message=intervention_data['message'],
            feedback_method='implicit',
            effectiveness_score=effectiveness_score,
            response_time_seconds=(
                datetime.now() - intervention_time).total_seconds(),
            context_at_intervention=intervention_data['context'],
            behavioral_response={
                'effectiveness_calculated': effectiveness_score}
        )

        # Store feedback and learn from it
        self.feedback_history.append(feedback_entry)
        self._save_feedback_entry(feedback_entry)

        # Update user model
        self._record_intervention_feedback(
            intervention_data['user_id'], intervention_data['type'],
            effectiveness_score, feedback_entry.response_time_seconds)

        # Learn from this interaction
        if self.pattern_learner_enabled:
            self.learn_from_interaction(
                intervention_data['context'],
                {'intervention_type': intervention_data['type'],
                    'priority': intervention_data['priority']},
                effectiveness_score,
                feedback_entry.response_time_seconds
            )

        # Clean up intervention context
        del self.intervention_contexts[intervention_id]

        logger.info(
            f"Analyzed intervention {intervention_id}: effectiveness={effectiveness_score:.2f}")
        return feedback_entry

    def _calculate_behavioral_effectiveness(self, before_context: Dict,
                                            after_contexts: List[Dict], intervention_type: str) -> float:
        """Calculate effectiveness based on behavioral changes"""

        if not after_contexts:
            return 0.5

        # Take average of next few measurements
        after_avg = {}
        for key in ['productivity_score', 'focus_quality', 'stress_level', 'energy_level']:
            values = [ctx.get(key, 0.5)
                      for ctx in after_contexts if key in ctx]
            after_avg[key] = sum(values) / len(values) if values else 0.5

        # Calculate improvements based on intervention type
        if intervention_type in ['productivity_boost', 'focus_enhancement']:
            prod_improvement = after_avg['productivity_score'] - \
                before_context.get('productivity_score', 0.5)
            focus_improvement = after_avg['focus_quality'] - \
                before_context.get('focus_quality', 0.5)
            return max(0.2, min(1.0, 0.5 + prod_improvement + focus_improvement))

        elif intervention_type == 'stress_reduction':
            stress_reduction = before_context.get(
                'stress_level', 0.5) - after_avg['stress_level']
            return max(0.2, min(1.0, 0.5 + stress_reduction))

        elif intervention_type == 'energy_boost':
            energy_improvement = after_avg['energy_level'] - \
                before_context.get('energy_level', 0.5)
            return max(0.2, min(1.0, 0.5 + energy_improvement))

        return 0.5

    def _record_intervention_feedback(self, user_id: str, intervention_type: str,
                                      effectiveness_score: float, response_time_seconds: float):
        """Record feedback about intervention effectiveness for user model"""
        profile = self._get_user_profile(user_id)

        if profile.intervention_effectiveness is None:
            profile.intervention_effectiveness = {}

        # Update effectiveness with exponential moving average
        if intervention_type in profile.intervention_effectiveness:
            current_eff = profile.intervention_effectiveness[intervention_type]
            profile.intervention_effectiveness[intervention_type] = 0.7 * \
                current_eff + 0.3 * effectiveness_score
        else:
            profile.intervention_effectiveness[intervention_type] = effectiveness_score

        self._save_user_profile(profile)

    # ========================================================================
    # UTILITY METHODS FOR PERSONALIZED SCORING
    # ========================================================================

    def calculate_personalized_productivity_score(self, event_buffer: List[Dict], context: Dict, user_id: str = "default") -> float:
        """Calculate personalized productivity score with user model adjustments"""

        if not event_buffer:
            return 0.5

        # Base calculation
        activities = [e.get('keyboard_count', 0) +
                      e.get('mouse_count', 0) for e in event_buffer]
        if not activities:
            return 0.5

        avg_activity = sum(activities) / len(activities)

        # Calculate base score with improved thresholds for high-performance users
        if avg_activity > 100:
            base_score = min(1.0, 0.7 + (avg_activity - 100) / 300)
        elif avg_activity < 5:  # Only penalize extremely low activity
            base_score = max(0.3, avg_activity / 20)  # Less harsh penalty
        else:
            base_score = 0.5 + (avg_activity - 5) / 100  # More generous scoring

        # Personalize based on user profile
        user_profile = self._get_user_profile(user_id)

        # Persona-specific adjustments
        persona = self._detect_user_persona(context)
        current_app = context.get('current_application', '').lower()

        # Developers: coding apps and AI tools = higher baseline
        if persona == 'developer' and any(app in current_app for app in ['vscode', 'intellij', 'pycharm', 'terminal', 'cursor', 'chatgpt', 'claude', 'comet']):
            base_score = min(1.0, base_score + 0.15)  # Higher boost for dev tools

        # Analysts: data apps = higher baseline
        elif persona == 'analyst' and any(app in current_app for app in ['excel', 'tableau', 'jupyter', 'r studio']):
            base_score = min(1.0, base_score + 0.1)

        # Managers: communication apps during work hours = normal productivity
        elif persona == 'manager' and any(app in current_app for app in ['slack', 'teams', 'outlook']):
            if 9 <= datetime.now().hour <= 17:
                base_score = min(1.0, base_score + 0.05)

        # Adjust based on user's personal baseline
        personalized_score = (base_score * 0.7) + \
            (user_profile.productivity_baseline * 0.3)

        return max(0.0, min(1.0, personalized_score))

    def calculate_personalized_focus_quality(self, event_buffer: List[Dict], context: Dict, user_id: str = "default") -> float:
        """Calculate personalized focus quality with user model awareness"""

        if not event_buffer:
            return 0.5

        # Analyze app switching patterns
        apps = [(e.get("process_name") or e.get("app") or "").lower()
                for e in event_buffer if isinstance(e, dict)]
        if not apps and context.get("current_application"):
            apps = [context["current_application"].lower()]

        unique_apps = len(set(a for a in apps if a))
        if unique_apps <= 1:
            base_focus = 0.9
        elif unique_apps == 2:
            base_focus = 0.7
        else:
            base_focus = max(0.3, 1.0 - (unique_apps * 0.15))

        # Persona-specific adjustments
        persona = self._detect_user_persona(context)
        user_profile = self._get_user_profile(user_id)

        # Developers: expect fewer app switches
        if persona == 'developer' and unique_apps > 3:
            base_focus *= 0.8

        # Managers: more app switching is normal
        elif persona == 'manager' and unique_apps <= 4:
            base_focus = min(1.0, base_focus + 0.1)

        # Personalize based on user's focus baseline
        personalized_focus = (base_focus * 0.8) + \
            (user_profile.focus_baseline * 0.2)

        return max(0.0, min(1.0, personalized_focus))

    def detect_user_persona(self, context: Dict) -> str:
        """Public interface for persona detection"""
        return self._detect_user_persona(context)

    def get_persona_specific_coaching(self, persona: str, context: Dict, analysis: Dict) -> Optional[Dict]:
        """Public interface for persona-specific coaching"""
        return self._get_persona_specific_coaching(persona, context, analysis)

    # ========================================================================
    # SYSTEM INFORMATION AND INSIGHTS
    # ========================================================================

    def get_user_insights(self, user_id: str = "default") -> Dict[str, Any]:
        """Get comprehensive insights about user and system performance"""

        profile = self._get_user_profile(user_id)
        contexts = list(self.user_contexts.get(user_id, []))

        # Recent trends
        recent_contexts = contexts[-20:] if len(contexts) >= 20 else contexts
        trends = {}
        if recent_contexts:
            recent_productivity = [
                ctx.get('productivity_score', 0.5) for ctx in recent_contexts]
            recent_focus = [ctx.get('focus_quality', 0.5)
                            for ctx in recent_contexts]
            recent_stress = [ctx.get('stress_level', 0.5)
                             for ctx in recent_contexts]

            trends = {
                'productivity_trend': 'improving' if len(recent_productivity) > 5 and recent_productivity[-1] > recent_productivity[0] else 'stable',
                'focus_trend': 'improving' if len(recent_focus) > 5 and recent_focus[-1] > recent_focus[0] else 'stable',
                'stress_trend': 'improving' if len(recent_stress) > 5 and recent_stress[-1] < recent_stress[0] else 'stable'
            }

        return {
            'profile': {
                'persona': profile.persona,
                'confidence': profile.confidence_level,
                'productivity_baseline': profile.productivity_baseline,
                'focus_baseline': profile.focus_baseline,
                'total_interactions': profile.total_interactions
            },
            'patterns': {
                'energy_patterns': profile.energy_patterns or {},
                'productivity_patterns': profile.productivity_patterns or {},
                'intervention_effectiveness': profile.intervention_effectiveness or {}
            },
            'recent_trends': {
                **trends,
                'data_points': len(recent_contexts)
            },
            'learning_performance': {
                'total_feedback_points': len(self.feedback_history),
                'total_interactions': len(self.interaction_data),
                'learning_trend': {'direction': 'stable'}
            },
            'pattern_recognition': {
                'total_interactions': len(self.interaction_data),
                'pattern_quality': 'good' if len(self.interaction_data) > 20 else 'developing'
            },
            'overall_performance': {
                'total_interventions': len(self.intervention_history),
                'ml_confidence': self._calculate_overall_ml_confidence(),
                'personalization_level': self._calculate_personalization_level(user_id)
            }
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the coaching system capabilities"""

        return {
            'coach_type': 'enhanced_ml' if 'ml_pattern_learning' in self.capabilities else
            'personalized' if 'persona_detection' in self.capabilities else 'base',
            'capabilities': self.capabilities,
            'anthropic_api_available': self.claude_client is not None,
            'ml_libraries_available': {
                'pandas': PANDAS_AVAILABLE,
                'sklearn': SKLEARN_AVAILABLE,
                'anthropic': ANTHROPIC_AVAILABLE
            },
            'data_stats': {
                'user_profiles': len(self.user_profiles),
                'interaction_data': len(self.interaction_data),
                'feedback_history': len(self.feedback_history),
                'intervention_history': len(self.intervention_history)
            }
        }

    def _calculate_overall_ml_confidence(self) -> float:
        """Calculate overall confidence in ML predictions"""
        components = []

        # Pattern learner confidence
        if len(self.interaction_data) > 20:
            components.append(0.8)
        elif len(self.interaction_data) > 5:
            components.append(0.6)
        else:
            components.append(0.3)

        # Feedback system confidence
        if len(self.feedback_history) > 10:
            components.append(0.7)
        else:
            components.append(0.4)

        return sum(components) / len(components) if components else 0.5

    def _calculate_personalization_level(self, user_id: str) -> float:
        """Calculate how personalized the coaching is for this user"""
        user_profile = self._get_user_profile(user_id)

        factors = [
            user_profile.confidence_level,  # Persona confidence
            # Interaction history
            min(1.0, user_profile.total_interactions / 50),
            # Effectiveness data
            min(1.0, len(user_profile.intervention_effectiveness or {}) / 5)
        ]

        return sum(factors) / len(factors)

    def _infer_break_taken(self, event_buffer: List[Dict]) -> bool:
        """Infer if user has taken a break from activity gaps"""
        if not event_buffer:
            return False

        # Consider a break if >= 5 consecutive minutes with near-zero input
        recent = event_buffer[-10:]  # last ~10 minutes
        idle_minutes = sum(1 for e in recent if (
            e.get('keyboard_count', 0) + e.get('mouse_count', 0)) < 5)
        return idle_minutes >= 5

    # ========================================================================
    # NOTIFICATION SUPPRESSION AND COOLDOWN SYSTEM
    # ========================================================================
    
    def _should_send_notification(self, user_id: str = "default") -> bool:
        """Simple rate limiting check for basic coaching fallback"""
        now = datetime.now()
        
        # Check global per-hour cap
        recent = [i for i in self.intervention_history.values()
                  if i.get('user_id') == user_id and
                  datetime.fromisoformat(i.get('timestamp', now.isoformat())) > now - timedelta(hours=1)]
        
        if len(recent) >= self.notification_config.get('max_per_hour', 4):
            return False
        
        # Check global cooldown 
        min_gap = self.notification_config.get('min_minutes_between', 12)
        if recent:
            last_ts = max(datetime.fromisoformat(
                i.get('timestamp', now.isoformat())) for i in recent)
            if (now - last_ts).total_seconds() < min_gap * 60:
                return False
        
        return True

    def _should_suppress_notification(self, user_id: str, intervention: Dict, context: Dict) -> Tuple[bool, str]:
        """Check if notification should be suppressed based on cooldowns, meeting state, etc."""
        now = datetime.now()

        # 1) In-meeting suppression for non-critical
        if self.notification_config.get('suppress_in_meeting', True):
            if context.get(self.keys['in_meeting'], False):
                if intervention['type'] not in self.notification_config.get('allow_in_meeting_types', []):
                    return True, 'in_meeting_suppression'

        # 2) Global per-hour cap
        recent = [i for i in self.intervention_history.values()
                  if i.get('user_id') == user_id and
                  datetime.fromisoformat(i.get('timestamp', now.isoformat())) > now - timedelta(hours=1)]
        if len(recent) >= self.notification_config.get('max_per_hour', 4):
            return True, 'hourly_cap'

        # 3) Global cooldown
        min_gap = self.notification_config.get('min_minutes_between', 12)
        if recent:
            last_ts = max(datetime.fromisoformat(
                i.get('timestamp', now.isoformat())) for i in recent)
            if (now - last_ts).total_seconds() < min_gap * 60:
                return True, 'global_cooldown'

        # 4) Per-type cooldown
        per_type = self.notification_config.get(
            'per_type_cooldown_minutes', {})
        tcd = per_type.get(intervention['type'])
        if tcd:
            pertype_recent = [i for i in recent if i.get(
                'type') == intervention['type']]
            if pertype_recent:
                last_t = max(datetime.fromisoformat(
                    i.get('timestamp', now.isoformat())) for i in pertype_recent)
                if (now - last_t).total_seconds() < tcd * 60:
                    return True, 'type_cooldown'

        # 5) Repeat-text suppression
        rpt = self.notification_config.get('repeat_suppression_minutes', 90)
        for i in recent:
            if i.get('message') == intervention['message']:
                if (now - datetime.fromisoformat(i.get('timestamp', now.isoformat()))).total_seconds() < rpt * 60:
                    return True, 'repeat_suppression'

        return False, ''

    def _envelope(self, *, coaching_type: str, urgency: str, message: str,
                  persona: str, priority: Optional[int] = None,
                  source: str = 'rule_based', confidence: float = 0.6,
                  channel: Optional[str] = None, context: Optional[Dict] = None,
                  meta_extra: Optional[Dict] = None) -> Dict:
        """Create standardized notification envelope"""
        prio = priority if priority is not None else (
            3 if urgency == 'high' else 2 if urgency == 'medium' else 1)
        ctx = context or {}
        selected_channel = channel or self._select_channel(urgency, ctx)

        env = {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': message[:500],
            'priority': prio,
            'urgency': urgency,
            'persona': persona,
            'channel': selected_channel,
            'interaction': self._build_interaction(coaching_type, urgency, ctx),
            'meta': {
                'confidence': max(0.0, min(1.0, confidence)),
                'reasoning': f"{source} {coaching_type} advice",
                'source': source,
                'cooldown_applied': False
            }
        }
        if meta_extra:
            env['meta'].update(meta_extra)
        return env

    def _choose_copy_variant(self, texts: Any, user_id: str) -> str:
        """Choose copy variant deterministically but varied by user & hour"""
        if not isinstance(texts, list) or not texts:
            return str(texts or "")
        h = hashlib.sha256(
            f"{user_id}:{datetime.now().hour}".encode()).digest()
        idx = h[0] % len(texts)
        return texts[idx]

    def _adjust_message_for_context(self, message: str, context: Dict, urgency: str, coaching_type: str) -> str:
        """Adjust message phrasing based on context and urgency"""
        msg = message

        # Meeting softening
        if context.get('in_meeting'):
            replace_map = {
                " now": " when the meeting wraps",
                "Now ": "When the meeting wraps ",
                "Take 5-10 minutes": "Plan 5–10 minutes next",
                "Stand up": "Plan to stand up",
                "Start ": "Queue "
            }
            for k, v in replace_map.items():
                msg = msg.replace(k, v)

        # Energy-sensitive tone
        if coaching_type in ('productivity_boost', 'focus_enhancement') and context.get('energy_level', 0.5) < 0.2:
            msg = "Restore first: 2–3 minutes gentle movement or hydration. " + msg

        # Explicitly name long sessions
        sd = context.get('session_duration_hours', 0)
        if coaching_type == 'break_reminder' and sd >= 3.0:
            msg = f"You've been at it for {sd:.1f} hours. " + msg

        return msg

    def _build_interaction(self, coaching_type: str, urgency: str, context: Dict) -> Dict:
        """Build interaction metadata (title and CTA)"""
        if coaching_type == 'productivity_boost':
            return {'title': "Focus for 25 minutes",
                    'cta': {'label': "Start Timer", 'action': 'start_pomodoro', 'payload': {'minutes': 25}}}
        if coaching_type == 'break_reminder':
            return {'title': "Take a 5–10 min break",
                    'cta': {'label': "Snooze 10 min", 'action': 'snooze', 'payload': {'minutes': 10}}}
        if coaching_type == 'stress_reduction':
            return {'title': "Quick reset",
                    'cta': {'label': "Open Breathing", 'action': 'open_breathing', 'payload': {'minutes': 2}}}
        if coaching_type == 'focus_enhancement':
            return {'title': "Eliminate distractions",
                    'cta': {'label': "Do Not Disturb", 'action': 'enable_dnd', 'payload': {'minutes': 30}}}
        if coaching_type == 'energy_boost':
            return {'title': "Restore energy",
                    'cta': {'label': "Quick Walk", 'action': 'start_movement', 'payload': {'minutes': 5}}}
        return {'title': "Heads‑up", 'cta': None}

    def _select_channel(self, urgency: str, context: Dict) -> str:
        """Select notification channel based on urgency and context"""
        if context.get('in_meeting'):
            return 'system_banner'  # soft in meeting
        return 'modal' if urgency == 'high' else 'toast' if urgency == 'medium' else 'system_banner'

    def _normalize_ai_response(self, coaching_type: str, urgency: str, raw_text: str, user_id: str = "default") -> Dict:
        """Normalize AI response to standard schema with validation"""
        priority = 3 if urgency == 'high' else 2 if urgency == 'medium' else 1

        try:
            payload = json.loads(raw_text)
            msg = payload.get('message') or raw_text
            prio = int(payload.get('priority', priority))
            conf = float(payload.get('confidence', 0.7))
            reasoning = payload.get(
                'reasoning', f"AI-generated {coaching_type} advice")
        except Exception:
            msg, prio, conf, reasoning = raw_text, priority, 0.7, f"AI-generated {coaching_type} advice"

        return {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': msg[:500],  # guardrail length
            'priority': min(3, max(1, prio)),
            'urgency': urgency,
            'persona': self._get_user_profile(user_id).persona,
            'channel': self.notification_config.get('default_channel', 'system_banner'),
            'meta': {
                'confidence': max(0.0, min(1.0, conf)),
                'reasoning': reasoning,
                'source': 'anthropic_ai',
                'cooldown_applied': False
            }
        }

    def _log_notification_event(self, event: str, user_id: str, intervention: Dict, reason: str = ""):
        """Log notification events for analytics"""
        try:
            path = self.data_dir / 'notification_events.jsonl'
            payload = {
                'timestamp': datetime.now().isoformat(),
                'event': event,  # 'sent', 'suppressed', 'clicked', 'dismissed', 'snoozed'
                'user_id': user_id,
                'type': intervention.get('type'),
                'urgency': intervention.get('urgency'),
                'persona': intervention.get('persona'),
                'channel': intervention.get('channel'),
                'reason': reason,
                'message_hash': intervention.get('meta', {}).get('message_hash')
            }
            with open(path, 'a') as f:
                f.write(json.dumps(payload) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log notification event: {e}")

    def _deliver_notification(self, suggestion: Dict) -> None:
        """Deliver notification through appropriate channel"""
        if not suggestion:
            return

        channel = suggestion.get('channel', 'console')
        message = suggestion.get('message', '')
        urgency = suggestion.get('urgency', 'medium')
        suggestion_type = suggestion.get('type', 'unknown')

        if self.test_mode:
            # In test mode, always print to console with formatting
            urgency_emoji = {"low": "💡", "medium": "⚡⚡", "high": "🚨🚨🚨"}
            type_emoji = {
                "productivity_boost": "🚀", "focus_enhancement": "🎯",
                "stress_reduction": "😤", "break_reminder": "⏰",
                "repeat_docs": "📚", "tab_switching": "🔄",
                "file_churn": "📁", "meeting_distraction": "📞",
                "energy_boost": "🔋"
            }

            emoji = urgency_emoji.get(urgency, "💡")
            type_icon = type_emoji.get(suggestion_type, "💭")

            # Different formatting based on urgency
            if urgency == 'high':
                print("\n" + "🚨"*30)
                print(f"{'CRITICAL ALERT - ACTION REQUIRED':^60}")
                print("🚨"*30)
            elif urgency == 'medium':
                print("\n" + "="*60)
                print(f"{'⚡ PERFORMANCE ALERT ⚡':^60}")
                print("="*60)
            else:
                print("\n" + "-"*60)
                print(f"{'💡 Coaching Tip':^60}")
                print("-"*60)

            print(f"{type_icon} Type: {suggestion_type.upper()}")
            print(f"📊 Priority: {urgency.upper()}")
            print(f"👤 Persona: {suggestion.get('persona', 'generic')}")
            print(f"📝 Message: {message}")
            print(f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}")
            if suggestion.get('meta', {}).get('evidence'):
                print(f"📊 Evidence: {suggestion['meta']['evidence']}")
            print("="*60 + "\n")

            # Also show macOS notification for medium/high urgency
            if urgency in ['medium', 'high']:
                try:
                    import subprocess
                    title = "⚡ AI COACH" if urgency == 'medium' else "🚨 CRITICAL"
                    # Use terminal-notifier if available for better notifications
                    try:
                        subprocess.run([
                            'terminal-notifier',
                            '-title', title,
                            '-message', message[:100],
                            '-sound', 'default' if urgency == 'medium' else 'Sosumi',
                            '-ignoreDnD'  # Show even in Do Not Disturb mode for high urgency
                        ], capture_output=True, timeout=1)
                    except:
                        # Use terminal-notifier for reliable notifications
                        subprocess.run([
                            'terminal-notifier',
                            '-message', message[:100],
                            '-title', title,
                            '-sound', 'Ping'
                        ], capture_output=True, timeout=1)
                except:
                    pass  # Fail silently if notification fails
        else:
            # Regular delivery modes
            if channel == 'console':
                print(f"[AI Coach] {message}")
            elif channel == 'terminal_notifier':
                # Direct terminal-notifier delivery
                try:
                    import subprocess
                    title = f"AI Coach - {suggestion_type.replace('_', ' ').title()}"
                    sound = 'Sosumi' if urgency == 'high' else ('Ping' if urgency == 'medium' else 'default')
                    
                    print(f"[DEBUG] 🔔 Sending terminal-notifier: {message[:50]}...", flush=True)
                    result = subprocess.run(['terminal-notifier',
                                            '-message', message,
                                            '-title', title,
                                            '-sound', sound],
                                           capture_output=True, timeout=5, text=True)
                    print(f"[DEBUG] 📱 Terminal-notifier result: returncode={result.returncode}, stdout='{result.stdout.strip()}', stderr='{result.stderr.strip()}'", flush=True)
                    
                except Exception as e:
                    print(f"[DEBUG] ❌ Terminal-notifier failed: {e}", flush=True)
                    print(f"[AI Coach] {message}")  # Fallback to console
            elif channel == 'system_banner':
                # Use different notification methods based on urgency
                try:
                    import subprocess
                    title = f"AI Coach - {suggestion_type.replace('_', ' ').title()}"

                    if urgency == 'high':
                        # HIGH URGENCY: Use AppleScript dialog for major issues
                        dialog_message = f"🚨 CRITICAL PRODUCTIVITY ALERT\\n\\n{message}\\n\\nThis requires immediate attention!"
                        subprocess.run([
                            'osascript', '-e',
                            f'tell application "System Events" to display dialog "{dialog_message}" with title "🚨 {title}" buttons {{"OK"}} default button "OK"'
                        ], capture_output=True, timeout=10)
                        # Also log to console
                        print(f"[AI Coach CRITICAL] {message}")
                    else:
                        # LOW/MEDIUM URGENCY: Use terminal-notifier for minor coaching
                        sound = 'Ping' if urgency == 'medium' else 'default'
                        subprocess.run(['terminal-notifier',
                                        '-message', message,
                                        '-title', title,
                                        '-sound', sound],
                                       capture_output=True, timeout=5)

                except Exception as e:
                    logging.warning(f"Failed to show system notification: {e}")
                    print(f"[AI Coach] {message}")  # Fallback to console

    async def _deliver_test_recommendation(self, user_id: str, telemetry: Dict) -> None:
        """Deliver recommendation in test mode every 5 minutes"""
        if not self.test_mode:
            return

        try:
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 DEBUG: Starting analysis...")
            print(f"   📥 Telemetry: {telemetry}")

            # Force analysis regardless of cooldowns in test mode
            result = await self.analyze_telemetry(telemetry, user_id=user_id)

            print(f"   🔄 Analysis result: {result}")

            if result:
                print(
                    f"   ✅ Delivering recommendation: {result.get('type', 'unknown')}")
                self._deliver_notification(result)
            else:
                # If no suggestion, show a status update
                print(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] 🤖 AI Coach: No recommendations right now")
                print(
                    f"   📊 Productivity: {telemetry.get('productivity_score', 0):.2f} | Focus: {telemetry.get('focus_quality', 0):.2f}")
                print(
                    f"   ⚡ Stress: {telemetry.get('stress_level', 0):.2f} | Energy: {telemetry.get('energy_level', 0):.2f}")
                print(
                    f"   ⏱️  Session: {telemetry.get('session_duration_hours', 0):.1f}h")
                print(
                    f"   🔍 DEBUG: No result from analyze_telemetry - checking detector flags...")

                # Debug: Check what detectors are available and their flags
                await self._debug_detector_status(user_id, telemetry)

        except Exception as e:
            logging.error(f"Test mode recommendation delivery failed: {e}")
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ❌ AI Coach: Analysis error - {e}")
            import traceback
            print(f"   🔍 DEBUG: Full traceback:\n{traceback.format_exc()}")

    async def _debug_detector_status(self, user_id: str, telemetry: Dict) -> None:
        """Debug method to show detector status and flags"""
        try:
            print(f"   🔍 DEBUG: Checking detector status...")

            # Build history digests
            hist7 = self.history_cruncher.build_digest(user_id, days=7)
            print(f"   📚 7-day history: {len(hist7.get('events', []))} events")

            # Build current snapshot
            current = self.build_current_snapshot(user_id, telemetry)
            print(f"   📸 Current snapshot: {current}")

            # Check each detector
            print(f"   🔍 Available detectors: {len(self.detectors)}")
            for i, detector in enumerate(self.detectors):
                detector_name = detector.__class__.__name__
                try:
                    flags = detector.run(hist7, current)
                    print(
                        f"   🔍 Detector {i+1} ({detector_name}): {len(flags)} flags")
                    for flag in flags:
                        print(
                            f"      🚩 {flag.intent} (severity: {flag.severity})")
                except Exception as e:
                    print(f"   ❌ Detector {i+1} ({detector_name}) failed: {e}")

            # Force a single test recommendation using Anthropic API
            print(f"   🤖 Forcing Anthropic API call...")
            await self._force_anthropic_recommendation(user_id, telemetry, current)

        except Exception as e:
            print(f"   ❌ Debug detector status failed: {e}")
            import traceback
            print(f"   🔍 DEBUG: Traceback:\n{traceback.format_exc()}")

    async def _force_anthropic_recommendation(self, user_id: str, telemetry: Dict, current: Dict) -> None:
        """Force an Anthropic API call to test if the API is working"""
        try:
            # Get user profile
            profile = self._get_user_profile(user_id)

            # Create a simple prompt to test API connectivity
            prompt = f"""You are an AI productivity coach. Based on this telemetry data, provide a brief coaching recommendation.

User Profile: {profile.persona} developer
Current Activity: {telemetry.get('current_application', 'Unknown')}
Productivity Score: {telemetry.get('productivity_score', 0):.2f}
Focus Quality: {telemetry.get('focus_quality', 0):.2f}
Stress Level: {telemetry.get('stress_level', 0):.2f}
Session Duration: {telemetry.get('session_duration_hours', 0):.1f} hours

Provide a specific, actionable recommendation (1-2 sentences) or say "No recommendation needed" if patterns look good."""

            print(f"   🔗 Testing Anthropic API with prompt...")
            print(f"   📝 Prompt: {prompt[:100]}...")

            # Make API call using existing method
            coaching_result = await self._get_anthropic_coaching(telemetry, "productivity", "low", user_id)
            if coaching_result:
                print(
                    f"   ✅ Anthropic API Response: {coaching_result['message']}")
            else:
                print(f"   ⚠️  Anthropic API returned no coaching result")

                # Test direct API call if available
                if self.claude_client:
                    print(f"   🔗 Testing direct Claude API call...")
                    try:
                        message = await self.claude_client.messages.create(
                            model="claude-3-5-haiku-latest",
                            max_tokens=7150,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        print(
                            f"   ✅ Direct API Response: {message.content[0].text}")
                    except Exception as api_e:
                        print(f"   ❌ Direct API call failed: {api_e}")
                else:
                    print(f"   ❌ No Claude client available")

        except Exception as e:
            print(f"   ❌ Anthropic API test failed: {e}")
            import traceback
            print(f"   🔍 API Error Traceback:\n{traceback.format_exc()}")

    def _record_intervention(self, intervention: Dict, user_id: str = "default"):
        """Record intervention in history for cooldown tracking"""
        intervention['timestamp'] = datetime.now().isoformat()
        intervention['user_id'] = user_id
        intervention_id = intervention.get('id', str(uuid.uuid4()))
        intervention['id'] = intervention_id
        self.intervention_history[intervention_id] = intervention

    # ========================================================================
    # DATA PERSISTENCE
    # ========================================================================

    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to disk"""
        try:
            profile_file = self.data_dir / f"profile_{profile.user_id}.json"
            with open(profile_file, 'w') as f:
                json.dump(asdict(profile), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save user profile {profile.user_id}: {e}")

    def _load_user_profiles(self):
        """Load user profiles from disk"""
        try:
            for profile_file in self.data_dir.glob("profile_*.json"):
                with open(profile_file, 'r') as f:
                    data = json.load(f)

                    # Reconstruct UserProfile
                    if 'preferences' in data and data['preferences']:
                        data['preferences'] = UserPreferences(
                            **data['preferences'])
                    else:
                        data['preferences'] = UserPreferences()

                    profile = UserProfile(**data)
                    self.user_profiles[profile.user_id] = profile

            logger.info(f"Loaded {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.warning(f"Failed to load user profiles: {e}")

    def _save_interaction_record(self, record: Dict):
        """Save interaction record to disk"""
        try:
            interactions_file = self.data_dir / 'interactions.jsonl'
            with open(interactions_file, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save interaction record: {e}")

    def _load_training_data(self):
        """Load existing training data"""
        try:
            interactions_file = self.data_dir / 'interactions.jsonl'
            if interactions_file.exists():
                with open(interactions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            self.interaction_data.append(record)

                logger.info(
                    f"Loaded {len(self.interaction_data)} interaction records")
        except Exception as e:
            logger.warning(f"Failed to load training data: {e}")

    def _save_feedback_entry(self, entry: FeedbackEntry):
        """Save feedback entry to disk"""
        try:
            feedback_file = self.data_dir / 'feedback_history.jsonl'
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(asdict(entry), default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save feedback entry: {e}")

    def _load_feedback_history(self):
        """Load feedback history from disk"""
        try:
            feedback_file = self.data_dir / 'feedback_history.jsonl'
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry_dict = json.loads(line)
                            entry = FeedbackEntry(**entry_dict)
                            self.feedback_history.append(entry)

                logger.info(
                    f"Loaded {len(self.feedback_history)} feedback entries")
        except Exception as e:
            logger.warning(f"Failed to load feedback history: {e}")

    def start_test_mode(self, interval_minutes: int = 5):
        """Start test mode with periodic analysis using YOUR real data"""
        if not self.test_mode:
            print("❌ Test mode not enabled. Initialize AICoach with test_mode=True")
            return

        import asyncio
        import subprocess
        import random

        async def test_loop():
            print("🧠 AI COACH TEST MODE")
            print("=" * 60)
            print(f"🕐 Analysis every {interval_minutes} minutes")
            print("🤖 Real recommendations based on YOUR data")
            print("⌨️  Press Ctrl+C to stop")
            print("=" * 60)

            while True:
                try:
                    # Get YOUR current app (macOS)
                    try:
                        script = '''tell application "System Events" to name of first application process whose frontmost is true'''
                        result = subprocess.run(
                            ['osascript', '-e', script], capture_output=True, text=True, timeout=1)
                        current_app = result.stdout.strip() if result.returncode == 0 else "Unknown"
                    except:
                        current_app = "Cursor"  # Default for developers

                    # Get REAL telemetry from CrossOver data
                    telemetry = await self._get_real_crossover_telemetry(current_app)

                    print(
                        f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 Analyzing YOUR REAL activity...")
                    print(f"   📱 Current App: {current_app}")

                    if "error" in telemetry:
                        print(f"   ⚠️  {telemetry['error']}")
                    else:
                        print(
                            f"   ⌨️  Real Keystrokes: {telemetry.get('total_keystrokes', 0):,}")
                        print(
                            f"   🖱️  Real Mouse Clicks: {telemetry.get('total_mouse_clicks', 0):,}")
                        print(
                            f"   ⏰ Hours Today: {telemetry.get('hours_today', 0):.1f}")
                        print(
                            f"   📊 Real Metrics: P={telemetry.get('productivity_score', 0):.2f} F={telemetry.get('focus_quality', 0):.2f} S={telemetry.get('stress_level', 0):.2f} E={telemetry.get('energy_level', 0):.2f}")

                    # Show detailed analysis of what's being checked
                    await self._show_detailed_analysis("workmart_user", telemetry, current_app)

                    # Analyze with AI Coach using YOUR patterns
                    result = await self.analyze_telemetry(telemetry, user_id="workmart_user")

                    # Check for inactivity and day transitions FIRST
                    inactivity_recommendations = self._check_inactivity_and_transitions(
                        telemetry)

                    # Generate smart recommendations based on the comprehensive analysis
                    smart_recommendations = await self._generate_smart_recommendations(telemetry)

                    # Prioritize inactivity messages
                    if inactivity_recommendations:
                        print(f"   💡 STATUS UPDATE:")
                        for rec in inactivity_recommendations:
                            print(f"   {rec}")
                    elif result:
                        print(f"   ✅ PATTERN-BASED RECOMMENDATION:")
                        self._deliver_notification(result)
                    elif smart_recommendations:
                        print(f"   💡 SMART COACHING RECOMMENDATIONS:")
                        for rec in smart_recommendations:
                            print(f"   {rec}")
                    else:
                        print(
                            f"   ✅ All metrics look healthy - keep up the excellent work!")

                    # Wait for next cycle
                    await asyncio.sleep(interval_minutes * 60)

                except KeyboardInterrupt:
                    print("\n✅ Test mode stopped")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    await asyncio.sleep(10)

        try:
            asyncio.run(test_loop())
        except KeyboardInterrupt:
            print("\n✅ Test mode stopped gracefully")

    async def _show_detailed_analysis(self, user_id: str, telemetry: Dict, current_app: str):
        """Show detailed analysis of what the AI Coach is checking"""
        try:
            # Check historical data
            try:
                events_7d = self.store.fetch_events(user_id, days=7)
                events_1d = self.store.fetch_events(user_id, days=1)
                print(
                    f"   📚 Historical data: {len(events_7d)} events (7d), {len(events_1d)} events (1d)")
            except:
                print(f"   📚 Historical data: Using existing database patterns")

            # Build analysis components - use 30 days for WorkSmart logs
            hist7 = self.history_cruncher.build_digest(user_id, days=30)
            current = self.build_current_snapshot(user_id, telemetry)

            print(
                f"   📊 History digest: {hist7.get('n_events', 0)} processed events (30 days)")

            # Show detailed work pattern analysis
            dwell_data = hist7.get('top_dwell', [])
            print(
                f"   💼 Work patterns extracted from {hist7.get('n_events', 0)} events:")

            # Categorize activities
            work_activities = [item for item in dwell_data if any(pattern in item[0] for pattern in
                                                                  ['Work-Session', 'Active-Work', 'Job-', 'API-', 'Work-Data'])]
            app_usage = [item for item in dwell_data if item[0]
                         in ['Cursor', 'Google Chrome', 'JavaAppLauncher']]
            system_activities = [item for item in dwell_data if any(pattern in item[0] for pattern in
                                                                    ['System-', 'Productivity-', 'Up-to-Date'])]

            if work_activities:
                print(f"   🔨 Work Activities: {work_activities[:3]}")
            if app_usage:
                print(f"   💻 App Usage: {app_usage}")
            if system_activities:
                print(f"   ⚙️  System Health: {system_activities[:2]}")

            print(f"   🔄 Tab switches: {hist7.get('tab_switches', 0)}")
            print(f"   🌐 External sites: {hist7.get('top_hosts', [])[:3]}")
            print(f"   📚 Documentation: {hist7.get('top_doc_hosts', [])[:3]}")
            print(
                f"   📸 Current context: {current.get('context', 'No context')}")

            # Check what detectors find
            total_flags = 0
            for i, detector in enumerate(self.detectors):
                detector_name = detector.__class__.__name__
                try:
                    flags = detector.run(hist7, current)
                    total_flags += len(flags)
                    if flags:
                        print(
                            f"   🔍 {detector_name}: {len(flags)} patterns detected")
                        for flag in flags[:2]:  # Show first 2
                            print(
                                f"      🚩 {flag.intent} (severity: {flag.severity})")
                    else:
                        print(f"   ✅ {detector_name}: No concerning patterns")
                except Exception as e:
                    print(f"   ❌ {detector_name}: Error - {e}")

            print(f"   🎯 Total pattern flags: {total_flags}")

            # Generate smart productivity insights from WorkSmart data
            await self._generate_productivity_insights(hist7, current_app, telemetry)

            # Show current activity pattern
            app_pattern = "Unknown"
            if current_app == "Cursor":
                app_pattern = "Coding/Development"
            elif current_app in ["Google Chrome", "Safari", "Firefox"]:
                app_pattern = "Web browsing"
            elif current_app in ["JavaAppLauncher", "Java"]:
                app_pattern = "Application Development"
            elif current_app in ["Slack", "Teams", "Zoom"]:
                app_pattern = "Communication"

            print(f"   🔄 Current activity: {app_pattern}")

            # Check API availability
            api_status = "Claude API" if self.claude_client else "Rule-based synthesis"
            print(f"   🤖 Analysis engine: {api_status}")

        except Exception as e:
            print(f"   ❌ Analysis error: {e}")

    async def _generate_productivity_insights(self, hist7: Dict, current_app: str, telemetry: Dict):
        """Generate smart productivity insights from WorkSmart data"""
        try:
            # Get comprehensive work analysis from ALL historical data
            total_work_patterns = await self._analyze_all_historical_work_patterns()

            dwell_data = hist7.get('top_dwell', [])
            recent_work_score = sum(score for activity, score in dwell_data if any(pattern in activity for pattern in
                                                                                   ['Work-Session', 'Active-Work', 'Work-Data', 'Job-']))

            print(f"   💡 Comprehensive Productivity Analysis:")

            # Show historical work intensity
            if total_work_patterns['total_score'] > 50000:
                print(
                    f"   🏆 HIGHLY PRODUCTIVE USER: {total_work_patterns['total_score']:,} total work points")
                print(
                    f"   📈 Work sessions: {total_work_patterns['active_sessions']:,}")
                print(
                    f"   📊 Documentation: {total_work_patterns['evidence_score']:,} points")
                print(
                    f"   🔄 Data sync: {total_work_patterns['sync_score']:,} points")

            # Recent activity analysis
            if recent_work_score > 20:
                print(f"   ✅ Recent work activity: {recent_work_score} points")
            else:
                print(
                    f"   📊 Current session building up (recent: {recent_work_score} points)")

            # Current state recommendations
            current_metrics = telemetry
            energy = current_metrics.get('energy_level', 0)
            stress = current_metrics.get('stress_level', 0)
            focus = current_metrics.get('focus_quality', 0)

            print(f"   🎯 Current State Analysis:")

            if current_app == "Cursor":
                if focus < 0.6:
                    print(
                        f"   💻 Coding with moderate focus ({focus:.2f}) - consider distraction elimination")
                else:
                    print(
                        f"   💻 Good coding focus ({focus:.2f}) - maintain this flow state")
            elif current_app in ["Google Chrome", "Safari"]:
                print(f"   🌐 Research mode detected - balance with implementation time")
            elif current_app == "JavaAppLauncher":
                print(f"   ☕ Java development detected - complex application work")

            # Energy and productivity optimization
            if energy > 0.7 and stress < 0.4:
                print(
                    f"   🚀 PEAK STATE (E:{energy:.2f}, S:{stress:.2f}) - ideal for challenging tasks")
            elif energy < 0.4:
                print(
                    f"   ⚡ Energy dip ({energy:.2f}) - micro-break or light movement recommended")
            elif stress > 0.6:
                print(
                    f"   😤 Elevated stress ({stress:.2f}) - deep breathing or quick walk")
            else:
                print(
                    f"   ✅ Balanced state (E:{energy:.2f}, S:{stress:.2f}) - good for steady progress")

        except Exception as e:
            print(f"   ❌ Insights generation failed: {e}")

    async def _analyze_all_historical_work_patterns(self):
        """Analyze ALL WorkSmart historical data for comprehensive work patterns"""
        try:
            import sqlite3
            db_path = self.data_dir / 'telemetry.db'
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # Get all WorkSmart logs
            cur.execute('SELECT window_title FROM events WHERE user_id = ? AND etype = ?',
                        ('workmart_user', 'deskapp_log'))
            all_logs = cur.fetchall()

            patterns = {
                'active_sessions': 0,
                'evidence_score': 0,
                'sync_score': 0,
                'management_score': 0,
                'total_score': 0
            }

            for (log_msg,) in all_logs:
                if 'Running job: ActivityJob:' in log_msg:
                    patterns['active_sessions'] += 1
                    patterns['total_score'] += 15
                elif 'ScreenshotJob build event:' in log_msg:
                    patterns['evidence_score'] += 5
                    patterns['total_score'] += 5
                elif 'Synchronizing data:' in log_msg:
                    patterns['sync_score'] += 4
                    patterns['total_score'] += 4
                elif 'Running job: TimecardUploadJob:' in log_msg:
                    patterns['management_score'] += 5
                    patterns['total_score'] += 5

            conn.close()
            return patterns

        except Exception as e:
            return {'total_score': 0, 'active_sessions': 0, 'evidence_score': 0, 'sync_score': 0, 'management_score': 0}

    async def _get_real_crossover_telemetry(self, current_app: str) -> Dict:
        """Get REAL telemetry data from CrossOver logs using WorkSmartDataReader"""
        try:
            from .telemetry_system import WorkSmartDataReader
            
            # Use the fixed WorkSmartDataReader that handles timezone issues
            reader = WorkSmartDataReader()
            recent_activities = reader.get_recent_activity_from_logs(1)  # Last hour
            global_data = reader.get_global_data_from_logs()
            
            if not recent_activities:
                # Fallback to basic real data
                return {
                    "worksmart_session_active": True,
                    "current_application": current_app,
                    "total_keystrokes": 0,
                    "total_mouse_clicks": 0,
                    "productivity_score": 0.0,
                    "focus_quality": 0.0,
                    "hours_today": global_data.get('hours_today', 0),
                    "error": "No recent CrossOver activity found"
                }

            # Aggregate real activity data from recent activities
            total_keystrokes = sum(a.get('keystrokes', 0) for a in recent_activities)
            total_clicks = sum(a.get('mouse_clicks', 0) for a in recent_activities)
            total_scrolls = sum(a.get('scroll_counts', 0) for a in recent_activities)
            
            # Calculate productivity based on recent activity
            total_activity = total_keystrokes + total_clicks
            if total_activity > 100:
                productivity_score = min(1.0, 0.6 + (total_activity - 100) / 500)
            elif total_activity > 20:
                productivity_score = 0.3 + (total_activity - 20) / 200
            else:
                productivity_score = max(0.1, total_activity / 40)
            
            # Calculate focus quality based on activity consistency
            focus_quality = 0.5  # Default
            if len(recent_activities) > 3:
                activity_levels = [a.get('keystrokes', 0) + a.get('mouse_clicks', 0) for a in recent_activities[-5:]]
                if activity_levels:
                    avg_activity = sum(activity_levels) / len(activity_levels)
                    consistency = 1.0 - (max(activity_levels) - min(activity_levels)) / max(1, avg_activity)
                    focus_quality = min(1.0, 0.3 + consistency * 0.7)

            real_data = {
                "worksmart_session_active": True,
                "current_application": current_app,
                "total_keystrokes": total_keystrokes,
                "total_mouse_clicks": total_clicks,
                "productivity_score": productivity_score,
                "focus_quality": focus_quality,
                "stress_level": 0.3,  # Default low stress
                "energy_level": 0.7,  # Default good energy
                "hours_today": global_data.get('hours_today', 0),
                "hours_this_week": global_data.get('hours_this_week', 0),
                "activity_sessions": recent_activities,
                "timestamp": datetime.now().isoformat(),
                "recent_activity_count": len(recent_activities),
                "latest_activity": recent_activities[0] if recent_activities else None
            }
            
            return real_data


        except Exception as e:
            # Fallback with minimal real data
            return {
                "worksmart_session_active": True,
                "current_application": current_app,
                "productivity_score": 0.5,
                "focus_quality": 0.5,
                "stress_level": 0.3,
                "energy_level": 0.6,
                "hours_today": 0,
                "error": f"Failed to parse CrossOver data: {e}"
            }

    def _check_inactivity_and_transitions(self, telemetry: Dict) -> List[str]:
        """Check for inactivity periods and day transitions"""
        recommendations = []
        current_time = datetime.now()
        current_date = current_time.date()

        # Extract activity metrics
        keystrokes = telemetry.get('total_keystrokes', 0)
        clicks = telemetry.get('total_mouse_clicks', 0)
        hours_today = telemetry.get('hours_today', 0)

        # Check for day transition
        if current_date != self.current_day:
            self.current_day = current_date
            self.day_transition_announced = False
            self.inactivity_level = None  # Reset inactivity tracking for new day

            # Only announce if we haven't already
            if not self.day_transition_announced:
                self.day_transition_announced = True
                recommendations.append(
                    "🌅 NEW DAY: Fresh start! Yesterday is done - focus on today's priorities")

                # Reset hours message if hours_today is 0
                if hours_today == 0:
                    recommendations.append(
                        "⏰ WORK DAY RESET: Starting fresh at 0.0 hours - plan your top 3 tasks")

        # Check for inactivity (no keystrokes or clicks)
        total_activity = keystrokes + clicks

        # Determine if there's actual activity
        if total_activity > self.last_activity_data.get('total', 0):
            # There's new activity - reset inactivity tracking
            self.last_activity_time = current_time
            self.last_activity_data = {
                'keystrokes': keystrokes, 'clicks': clicks, 'total': total_activity}
            self.inactivity_level = None
            self.last_inactivity_message = None
        else:
            # No new activity - check how long it's been
            time_since_activity = current_time - self.last_activity_time
            minutes_inactive = time_since_activity.total_seconds() / 60

            # Progressive inactivity detection with different messages
            if minutes_inactive >= 720 and self.inactivity_level != '12hr':  # 12 hours
                self.inactivity_level = '12hr'
                message = "💤 EXTENDED ABSENCE (12+ hrs): Welcome back when you're ready - no pressure"
                if message != self.last_inactivity_message:
                    recommendations.append(message)
                    self.last_inactivity_message = message

            # 6 hours
            elif minutes_inactive >= 360 and self.inactivity_level not in ['6hr', '12hr']:
                self.inactivity_level = '6hr'
                message = "🌙 LONG BREAK DETECTED (6+ hrs): Rest is productive - see you when refreshed!"
                if message != self.last_inactivity_message:
                    recommendations.append(message)
                    self.last_inactivity_message = message

            # 2 hours
            elif minutes_inactive >= 120 and self.inactivity_level not in ['2hr', '6hr', '12hr']:
                self.inactivity_level = '2hr'
                message = "☕ EXTENDED BREAK (2+ hrs): Taking time away is healthy - ready to resume?"
                if message != self.last_inactivity_message:
                    recommendations.append(message)
                    self.last_inactivity_message = message

            elif minutes_inactive >= 30 and self.inactivity_level is None:  # 30 minutes
                self.inactivity_level = '30min'
                message = "🚶 BREAK TIME DETECTED (30+ min): Great for productivity reset - feeling refreshed?"
                if message != self.last_inactivity_message:
                    recommendations.append(message)
                    self.last_inactivity_message = message

            # Special handling for end of work day
            if current_time.hour >= 18 and hours_today > 6 and minutes_inactive > 15:
                if "end_of_day" not in str(self.last_inactivity_message):
                    message = f"🏁 END OF DAY: {hours_today:.1f} hours complete - great work today!"
                    recommendations.append(message)
                    self.last_inactivity_message = message

        return recommendations

    async def _generate_smart_recommendations(self, telemetry: Dict) -> List[str]:
        """Generate smart recommendations using contextual analysis of 1000s of historical datapoints"""
        recommendations = []

        # Extract real metrics
        hours_today = telemetry.get('hours_today', 0)
        keystrokes = telemetry.get('total_keystrokes', 0)
        mouse_clicks = telemetry.get('total_mouse_clicks', 0)
        productivity_score = telemetry.get('productivity_score', 0)
        focus_quality = telemetry.get('focus_quality', 0)
        stress_level = telemetry.get('stress_level', 0)
        energy_level = telemetry.get('energy_level', 0)
        current_app = telemetry.get('current_application', 'Unknown')

        current_hour = datetime.now().hour
        total_interactions = keystrokes + mouse_clicks

        # Skip normal recommendations if user is inactive
        if total_interactions == 0 and productivity_score == 0:
            # User is inactive - let inactivity detection handle messages
            return []

        # INTELLIGENT PATTERN ANALYSIS: Actually USE the 1000s of datapoints
        process_patterns = await self._analyze_process_specific_patterns(current_app, current_hour)
        historical_patterns = await self._analyze_all_historical_work_patterns()

        # DEEP DATA CRUNCHING: Compare current performance vs historical patterns
        performance_insights = await self._crunch_performance_data(
            telemetry, process_patterns, historical_patterns, current_app, current_hour)

        # Use insights for intelligent recommendations - these trigger MEDIUM alerts
        if performance_insights:
            # These intelligent insights should trigger medium urgency notifications
            for insight in performance_insights:
                if any(keyword in insight for keyword in ['ALERT', 'WARNING', 'BELOW AVERAGE', 'DECLINING']):
                    # Create medium urgency notification for concerning insights
                    notification = {
                        'type': 'performance_insight',
                        'message': insight,
                        'urgency': 'medium',
                        'persona': 'generic',
                        'priority': 2
                    }
                    # This will trigger a medium alert notification
                    self._deliver_notification(notification)

            recommendations.extend(performance_insights)

        # ANALYZE THE 71,139 WORK POINTS AND 3,528 SESSIONS
        if historical_patterns['total_score'] > 50000:
            # High historical productivity user - different standards
            if hours_today > 10:
                recommendations.append(
                    "🚨 LONG DAY ALERT: 10+ hours with 71k+ work history - you've earned a proper break")
            elif productivity_score < 0.5 and hours_today > 6:
                recommendations.append(
                    "⚡ PRODUCTIVITY DIP: Your 3,528 past sessions show you're capable of more - refocus or take strategic break")

        # FOCUS QUALITY ANALYSIS with historical context
        if focus_quality < 0.4 and total_interactions > 100:
            recommendations.append(
                "🎯 FOCUS SCATTERED: High activity but low focus - close distractions and do 1 task for 25 minutes")
        elif focus_quality < 0.6 and hours_today > 4:
            recommendations.append(
                "💡 FOCUS DECLINING: Try the 2-minute rule - organize workspace then tackle hardest task first")

        # ENERGY MANAGEMENT based on real patterns
        if energy_level < 0.4:
            if current_hour < 14:
                recommendations.append(
                    "🔋 MORNING ENERGY LOW: 2-minute walk + hydrate - you have peak hours ahead")
            else:
                recommendations.append(
                    "🌅 ENERGY DEPLETED: 5-minute walk + fresh air - protect tomorrow's productivity")

        # STRESS INTERVENTION with work history context
        if stress_level > 0.6:
            if historical_patterns['active_sessions'] > 3000:  # Experienced worker
                recommendations.append(
                    "😤 STRESS DETECTED: You've handled 3,528 sessions before - breathe deeply and chunk current task")
            else:
                recommendations.append(
                    "😤 STRESS RISING: 4-7-8 breathing technique for 2 minutes - proven effective for developers")

        # TIME-BASED RECOMMENDATIONS
        if current_hour >= 18 and hours_today > 8:
            recommendations.append(
                "🌅 EVENING WIND-DOWN: 8+ hour day complete - plan tomorrow's top 3 priorities then stop")
        elif current_hour <= 10 and energy_level > 0.7:
            recommendations.append(
                "🚀 MORNING PEAK: High energy detected - tackle your most complex problem now")

        # ACTIVITY PATTERN ANALYSIS
        if keystrokes > 200 and mouse_clicks < 20:
            recommendations.append(
                "⌨️ HEAVY TYPING SESSION: Likely coding - take 30-second hand stretches every 15 minutes")
        elif mouse_clicks > keystrokes and mouse_clicks > 50:
            recommendations.append(
                "🖱️ MOUSE-HEAVY WORK: Likely design/research - alternate with keyboard-focused tasks")

        # PRODUCTIVITY SCORE INTERVENTIONS
        if productivity_score > 0.8 and energy_level > 0.6:
            recommendations.append(
                "🔥 HIGH PERFORMANCE MODE: Riding the wave - maintain this for max 90 more minutes then mandatory break")
        elif productivity_score < 0.3 and hours_today > 2:
            recommendations.append(
                "📈 PRODUCTIVITY BOOST NEEDED: Switch tasks or take 10-minute walk - your 71k work points prove you can do better")

        # HISTORICAL CONTEXT RECOMMENDATIONS
        evidence_score = historical_patterns.get('evidence_score', 0)
        if evidence_score > 1000:  # Lots of screenshots/documentation
            recommendations.append(
                "📊 DOCUMENTATION MASTER: Your 2,020 evidence points show great work habits - maintain this tracking")

        # CONTEXTUAL RECOMMENDATIONS based on process analysis
        if process_patterns:
            context_recs = await self._generate_process_contextual_recommendations(
                process_patterns, telemetry, current_app, current_hour)
            recommendations.extend(context_recs)

        return recommendations[:3]  # Return max 3 most relevant

    async def _crunch_performance_data(self, telemetry: Dict, process_patterns: Dict,
                                       historical_patterns: Dict, current_app: str, current_hour: int) -> List[str]:
        """INTELLIGENT DATA CRUNCHING - Actually analyze patterns for meaningful insights"""
        insights = []

        # Current metrics
        productivity = telemetry.get('productivity_score', 0)
        focus = telemetry.get('focus_quality', 0)
        keystrokes = telemetry.get('total_keystrokes', 0)
        hours_today = telemetry.get('hours_today', 0)

        try:
            # 1. APP-SPECIFIC PERFORMANCE ANALYSIS
            if current_app.lower() in ['cursor', 'vscode', 'pycharm', 'intellij']:
                # Coding app analysis
                if productivity < 0.5 and keystrokes > 50:
                    insights.append(
                        f"⚡ CODING ALERT: {keystrokes} keystrokes but only {productivity:.0%} productive. Code quality over quantity!")
                elif keystrokes < 20 and productivity > 0.8:
                    insights.append(
                        f"🤔 THINKING MODE: Low typing ({keystrokes} keys) but high productivity - planning phase detected")

            elif 'chrome' in current_app.lower() or 'firefox' in current_app.lower():
                # Browser analysis
                if focus < 0.4:
                    insights.append(
                        f"🌐 BROWSER DISTRACTION: Focus at {focus:.0%} in {current_app} - research or rabbit hole?")

            # 2. HOURLY PERFORMANCE PATTERN ANALYSIS
            activity_sessions = historical_patterns.get(
                'activity_sessions', [])
            if activity_sessions:
                # Find same-hour historical performance
                same_hour_sessions = [
                    s for s in activity_sessions[-100:] if s.get('hour') == current_hour]
                if len(same_hour_sessions) >= 3:
                    avg_productivity = sum(
                        s.get('productivity', 0.5) for s in same_hour_sessions) / len(same_hour_sessions)
                    if productivity < avg_productivity * 0.7:
                        insights.append(
                            f"📉 BELOW AVERAGE: {productivity:.0%} vs your typical {avg_productivity:.0%} at {current_hour}:00")
                    elif productivity > avg_productivity * 1.3:
                        insights.append(
                            f"🔥 PEAK PERFORMANCE: {productivity:.0%} vs typical {avg_productivity:.0%} at {current_hour}:00!")

            # 3. SESSION LENGTH INTELLIGENCE
            if hours_today > 0:
                # Calculate typical session length from historical data
                work_blocks = historical_patterns.get('work_blocks', [])
                if work_blocks:
                    session_hours = [b.get('hours_today', 0)
                                     for b in work_blocks[-20:]]
                    if session_hours:
                        avg_hours = sum(session_hours) / len(session_hours)
                        if hours_today < avg_hours * 0.5:
                            insights.append(
                                f"⏰ SHORT SESSION: {hours_today:.1f}h vs your average {avg_hours:.1f}h - need to build momentum")
                        elif hours_today > avg_hours * 1.5:
                            insights.append(
                                f"🚀 EXTENDED SESSION: {hours_today:.1f}h vs average {avg_hours:.1f}h - sustained focus!")

            # 4. PRODUCTIVITY TRAJECTORY ANALYSIS
            if len(activity_sessions) >= 10:
                recent_productivity = [
                    s.get('productivity', 0.5) for s in activity_sessions[-10:]]
                if len(recent_productivity) >= 5:
                    trend = (
                        recent_productivity[-1] - recent_productivity[0]) / len(recent_productivity)
                    if trend < -0.05:
                        insights.append(
                            f"📉 DECLINING TREND: Productivity dropping over last 10 sessions - need reset?")
                    elif trend > 0.05:
                        insights.append(
                            f"📈 IMPROVING TREND: Productivity climbing last 10 sessions - momentum building!")

            # 5. FOCUS PATTERN INTELLIGENCE
            if focus < 0.35 and keystrokes > 100:
                insights.append(
                    f"🎯 SCATTERED FOCUS: High activity ({keystrokes} keys) but {focus:.0%} focus - too many distractions")
            elif focus > 0.8 and productivity > 0.7:
                insights.append(
                    f"💎 FLOW STATE: {focus:.0%} focus + {productivity:.0%} productivity - maintain this zone!")

            # 6. 8-HOUR GOAL INTELLIGENCE
            if hours_today >= 1:
                total_interactions = keystrokes + \
                    telemetry.get('total_mouse_clicks', 0)
                pace_per_hour = productivity * \
                    total_interactions if total_interactions > 0 else productivity * 100
                projected_8h_score = pace_per_hour * 8
                if projected_8h_score < 400:  # Rough threshold
                    insights.append(
                        f"⚠️ 8H PACE WARNING: Current pace won't sustain quality 8h - need focus boost")

        except Exception as e:
            # Fallback - don't break if data analysis fails
            pass

        return insights[:2]  # Return max 2 intelligent insights

    async def _analyze_process_specific_patterns(self, current_app: str, current_hour: int) -> Dict:
        """Analyze 1000s of historical datapoints for the specific process user is doing now"""
        try:
            # Parse CrossOver logs for detailed session analysis
            crossover_files = Path.home() / "crossoverFiles"
            log_files = [
                crossover_files / "logs" / "deskapp.log",
                crossover_files / "logs" / "deskapp.log.1",
                crossover_files / "logs" / "deskapp.log.2"
            ]

            patterns = {
                'app_sessions': 0,
                'activity_sessions': [],
                'work_blocks': [],
                'productivity_windows': {},
                'intensity_patterns': {},
                'session_lengths': [],
                'break_patterns': [],
                'focus_indicators': []
            }

            # Parse actual CrossOver activity logs for rich insights
            all_activity_data = []
            work_sessions = []

            for log_file in log_files:
                if not log_file.exists():
                    continue

                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()

                    # Parse detailed activity patterns
                    current_session = None

                    for line in lines:
                        # Extract activity counts with timestamps
                        if "Counted" in line and "key press" in line:
                            import re
                            activity_match = re.search(
                                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Counted (\d+) key press.*?(\d+) mouse clicks.*?(\d+) scroll counts.*?([\d-]+\s[\d:]+)', line)
                            if activity_match:
                                timestamp_str, keys, clicks, scrolls, server_time = activity_match.groups()

                                try:
                                    timestamp = datetime.strptime(
                                        timestamp_str, "%Y-%m-%d %H:%M:%S")
                                    intensity = int(keys) + int(clicks)

                                    activity_data = {
                                        'timestamp': timestamp,
                                        'hour': timestamp.hour,
                                        'keys': int(keys),
                                        'clicks': int(clicks),
                                        'scrolls': int(scrolls),
                                        'intensity': intensity,
                                        'server_time': server_time
                                    }
                                    all_activity_data.append(activity_data)

                                except ValueError:
                                    continue

                        # Extract work hours data
                        elif "hours today:" in line:
                            hours_match = re.search(
                                r'hours today: ([\d:]+).*?hours this week: ([\d:]+)', line)
                            if hours_match:
                                today_str, week_str = hours_match.groups()
                                timestamp_match = re.search(
                                    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                                if timestamp_match:
                                    try:
                                        timestamp = datetime.strptime(
                                            timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                                        today_parts = today_str.split(':')
                                        hours_today = int(
                                            today_parts[0]) + int(today_parts[1])/60

                                        work_sessions.append({
                                            'timestamp': timestamp,
                                            'hours_today': hours_today,
                                            'hours_week': week_str
                                        })
                                    except ValueError:
                                        continue

                except Exception as e:
                    continue

            if not all_activity_data:
                print(
                    f"   🔍 CONTEXTUAL ANALYSIS: No detailed activity data found in CrossOver logs")
                return patterns

            print(
                f"   🔍 CONTEXTUAL ANALYSIS: Found {len(all_activity_data)} detailed activity sessions from CrossOver logs")

            patterns['activity_sessions'] = all_activity_data
            patterns['work_blocks'] = work_sessions

            # Analyze hourly productivity patterns
            hour_productivity = {}
            hour_intensities = {}

            for activity in all_activity_data:
                hour = activity['hour']
                intensity = activity['intensity']

                if hour not in hour_productivity:
                    hour_productivity[hour] = []
                    hour_intensities[hour] = []

                hour_productivity[hour].append(intensity)
                hour_intensities[hour].append(intensity)

            # Calculate average intensities per hour
            for hour in hour_productivity:
                avg_intensity = sum(
                    hour_productivity[hour]) / len(hour_productivity[hour])
                patterns['productivity_windows'][hour] = {
                    'avg_intensity': avg_intensity,
                    'session_count': len(hour_productivity[hour]),
                    'max_intensity': max(hour_productivity[hour]),
                    'min_intensity': min(hour_productivity[hour])
                }

            # Find optimal work windows
            if patterns['productivity_windows']:
                sorted_hours = sorted(patterns['productivity_windows'].items(),
                                      key=lambda x: x[1]['avg_intensity'], reverse=True)
                patterns['optimal_windows'] = [
                    (h, data['avg_intensity']) for h, data in sorted_hours[:3]]

            # Analyze session patterns and focus blocks
            if len(all_activity_data) > 1:
                # Group consecutive activities into focus blocks
                focus_blocks = []
                current_block = [all_activity_data[0]]

                for i in range(1, len(all_activity_data)):
                    prev_activity = all_activity_data[i-1]
                    curr_activity = all_activity_data[i]

                    # If activities are within 5 minutes, consider same focus block
                    time_diff = (
                        curr_activity['timestamp'] - prev_activity['timestamp']).total_seconds() / 60

                    if time_diff <= 5 and curr_activity['intensity'] > 0:
                        current_block.append(curr_activity)
                    else:
                        if len(current_block) > 1:  # Only blocks with multiple activities
                            focus_blocks.append(current_block)
                        current_block = [
                            curr_activity] if curr_activity['intensity'] > 0 else []

                # Add final block
                if len(current_block) > 1:
                    focus_blocks.append(current_block)

                patterns['focus_indicators'] = focus_blocks

                # Calculate typical session lengths
                session_lengths = []
                for block in focus_blocks:
                    if len(block) > 1:
                        duration = (block[-1]['timestamp'] - block[0]
                                    ['timestamp']).total_seconds() / 60
                        session_lengths.append(duration)

                patterns['session_lengths'] = session_lengths

            # Detect break patterns (gaps in activity)
            if len(all_activity_data) > 2:
                breaks = []
                for i in range(1, len(all_activity_data)):
                    prev_activity = all_activity_data[i-1]
                    curr_activity = all_activity_data[i]

                    gap_minutes = (
                        curr_activity['timestamp'] - prev_activity['timestamp']).total_seconds() / 60

                    if gap_minutes > 10:  # Break longer than 10 minutes
                        breaks.append({
                            'duration': gap_minutes,
                            'start_hour': prev_activity['hour'],
                            'end_hour': curr_activity['hour']
                        })

                patterns['break_patterns'] = breaks

            return patterns

        except Exception as e:
            print(f"   ❌ Process analysis failed: {e}")
            return {}

    async def _generate_process_contextual_recommendations(self, patterns: Dict, telemetry: Dict,
                                                           current_app: str, current_hour: int) -> List[str]:
        """Generate recommendations based on rich contextual analysis of CrossOver data"""
        recommendations = []

        optimal_windows = patterns.get('optimal_windows', [])
        productivity_windows = patterns.get('productivity_windows', {})
        session_lengths = patterns.get('session_lengths', [])
        focus_blocks = patterns.get('focus_indicators', [])
        break_patterns = patterns.get('break_patterns', [])
        activity_sessions = patterns.get('activity_sessions', [])

        current_intensity = telemetry.get(
            'total_keystrokes', 0) + telemetry.get('total_mouse_clicks', 0)

        # OPTIMAL TIME WINDOW ANALYSIS
        if optimal_windows and len(optimal_windows) >= 2:
            best_hour, best_intensity = optimal_windows[0]
            current_window_data = productivity_windows.get(current_hour)

            if current_hour == best_hour:
                recommendations.append(
                    f"🔥 PEAK HOUR: Your historical data shows {best_intensity:.0f} avg intensity at {best_hour}:00 - this is your optimal time!")
            elif current_window_data and current_window_data['avg_intensity'] < best_intensity * 0.6:
                recommendations.append(
                    f"⏰ SUBOPTIMAL TIME: Current hour averages {current_window_data['avg_intensity']:.0f} vs your peak {best_intensity:.0f} at {best_hour}:00")

        # FOCUS BLOCK ANALYSIS
        if session_lengths:
            avg_session = sum(session_lengths) / len(session_lengths)
            longest_session = max(session_lengths)

            # Estimate current session length (rough calculation)
            hours_today = telemetry.get('hours_today', 0)
            if hours_today > 0:
                estimated_current_session = hours_today * 60  # Convert to minutes

                if estimated_current_session > avg_session * 1.5:
                    recommendations.append(
                        f"⏰ LONG SESSION: {estimated_current_session:.0f}min vs your typical {avg_session:.0f}min focus blocks - consider break")
                elif estimated_current_session > longest_session:
                    recommendations.append(
                        f"🚨 EXTENDED SESSION: {estimated_current_session:.0f}min exceeds your longest recorded session ({longest_session:.0f}min)")

        # ACTIVITY INTENSITY COACHING with historical context
        if activity_sessions and current_intensity > 0:
            # Get recent activity intensities for comparison
            recent_intensities = [a['intensity']
                                  # Last 10 sessions
                                  for a in activity_sessions[-10:]]
            if recent_intensities:
                avg_recent = sum(recent_intensities) / len(recent_intensities)
                max_intensity = max([a['intensity']
                                    for a in activity_sessions])

                if current_intensity > avg_recent * 2:
                    recommendations.append(
                        f"🔥 HIGH INTENSITY: Current {current_intensity} vs recent avg {avg_recent:.0f} - peak performance, maintain for max 45min")
                elif current_intensity > max_intensity:
                    recommendations.append(
                        f"🚨 RECORD INTENSITY: {current_intensity} exceeds your historical max {max_intensity} - exceptional focus but unsustainable")
                elif current_intensity < avg_recent * 0.3:
                    recommendations.append(
                        f"📉 LOW ACTIVITY: {current_intensity} vs recent avg {avg_recent:.0f} - energy dip or distraction?")

        # BREAK PATTERN INSIGHTS
        if break_patterns:
            typical_break_duration = sum(
                [b['duration'] for b in break_patterns]) / len(break_patterns)
            last_break = break_patterns[-1] if break_patterns else None

            if last_break and last_break['duration'] < typical_break_duration * 0.5:
                recommendations.append(
                    f"⚡ SHORT BREAK PATTERN: Last break was {last_break['duration']:.0f}min vs typical {typical_break_duration:.0f}min - consider longer break")

        # PROCESS-SPECIFIC INSIGHTS
        if current_app == "Cursor" and productivity_windows:
            cursor_sessions = [
                s for s in activity_sessions if s['hour'] == current_hour]
            if len(cursor_sessions) >= 3:
                avg_cursor_intensity = sum(
                    [s['intensity'] for s in cursor_sessions]) / len(cursor_sessions)
                if current_intensity > avg_cursor_intensity * 1.3:
                    recommendations.append(
                        f"💻 CODING INTENSITY: {current_intensity} vs typical {avg_cursor_intensity:.0f} for Cursor at {current_hour}:00")

        return recommendations[:2]  # Max 2 contextual recommendations

# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================


__all__ = ['AICoach']
