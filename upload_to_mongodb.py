from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import toml

warnings.filterwarnings("ignore")

# Config 

DB_NAME = "kayfa_analytics"
DATA_DIR = Path("data")

# Helpers

def get_uri() -> str:
    """Resolve MONGO_URI from secrets.toml or environment."""
    # Try secrets.toml first
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        cfg = toml.load(secrets_path)
        if "MONGO_URI" in cfg:
            return cfg["MONGO_URI"]

    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        sys.exit(
            "ERROR: MONGO_URI not found. "
            "Add it to .streamlit/secrets.toml or set the MONGO_URI env variable."
        )
    return uri


def upload(db, name: str, df: pd.DataFrame) -> None:
    """Drop-and-replace a collection with the given DataFrame."""
    if df.empty:
        print(f"  [SKIP] {name} — empty DataFrame")
        return

    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        elif hasattr(df[col], "dt"):
            pass  # already handled

    docs = df.to_dict(orient="records")
    coll = db[name]
    coll.drop()
    coll.insert_many(docs)
    print(f"  [OK]   {name}  — {len(docs):,} docs")


# Load raw files 
def load_raw():
    print("\n── Loading raw files ─────────────────────────────────────────────────")
    courses     = pd.read_csv(DATA_DIR / "courses_clean.csv")
    groups      = pd.read_csv(DATA_DIR / "groups_cleaned.csv")
    students    = pd.read_csv(DATA_DIR / "students_clean.csv")
    grades      = pd.read_csv(DATA_DIR / "grades_clean.csv")
    attendance  = pd.read_excel(DATA_DIR / "attendance_clean.xlsx")
    concepts    = pd.read_csv(DATA_DIR / "concepts_performance_clean.csv")
    engagement  = pd.read_csv(DATA_DIR / "engagement_events_clean.csv")
    submissions = pd.read_csv(DATA_DIR / "assignment_submissions_clean.csv")

    # Parse types 
    students["age"]             = pd.to_numeric(students["age"], errors="coerce")
    students["enrollment_date"] = pd.to_datetime(students["enrollment_date"], errors="coerce")
    grades["score"]             = pd.to_numeric(grades["score"], errors="coerce")
    grades["date"]              = pd.to_datetime(grades["date"], errors="coerce")
    attendance["session_datetime"] = pd.to_datetime(attendance["session_datetime"], errors="coerce")
    engagement["event_datetime"]   = pd.to_datetime(engagement["event_datetime"], errors="coerce")
    engagement["duration_seconds"] = pd.to_numeric(engagement["duration_seconds"], errors="coerce")
    submissions["deadline"]     = pd.to_datetime(submissions["deadline"], errors="coerce")
    submissions["submitted_at"] = pd.to_datetime(submissions["submitted_at"], errors="coerce")
    concepts["score_pct"]  = pd.to_numeric(concepts["score_pct"], errors="coerce")
    concepts["timestamp"]  = pd.to_datetime(concepts["timestamp"], errors="coerce")
    if "Unnamed: 0" in submissions.columns:
        submissions.drop(columns=["Unnamed: 0"], inplace=True)
    engagement.loc[~engagement["duration_seconds"].between(0, 7200), "duration_seconds"] = np.nan


    # Merge group/course info into students 
    students = students.merge(
        groups[["group_id", "course_id", "instructor", "group_name"]],
        on="group_id", how="left"
    )
    students = students.merge(
        courses[["course_id", "course_name"]],
        on="course_id", how="left"
    )

    print("   Raw files loaded OK")
    return courses, groups, students, grades, attendance, concepts, engagement, submissions



# Aggregations 
def build_agg_student_risk(students, attendance, grades):
    student_att   = attendance.groupby("student_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    student_grade = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    df = (
        students[["student_id", "group_name", "course_name"]]
        .merge(student_att, on="student_id", how="left")
        .merge(student_grade, on="student_id", how="left")
    )
    df["is_at_risk"] = (
        (df["att_rate"].fillna(0) < 60) | (df["avg_grade"].fillna(0) < 55)
    )
    return df


def build_agg_concept_fail(concepts):
    return concepts.groupby("concept_name")["mastery_status"].apply(
        lambda x: (x == "failed").mean() * 100
    ).reset_index(name="fail_rate")


def build_agg_group_att(attendance, groups, courses):
    df = attendance.groupby("group_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    df = df.merge(groups[["group_id", "group_name", "course_id"]], on="group_id", how="left")
    df = df.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    return df


def build_agg_group_health(students, groups):
    true_group_size = students.groupby("group_id").size().reset_index(name="true_count")
    df = groups[["group_id", "group_name", "stated_num_students"]].merge(
        true_group_size, on="group_id", how="left"
    ).fillna({"true_count": 0})
    df["size_gap"]     = df["stated_num_students"] - df["true_count"]
    df["size_gap_abs"] = df["size_gap"].abs() + 1
    return df


def build_agg_monthly_att(attendance):
    df = attendance.copy()
    df["month"] = df["session_datetime"].dt.to_period("M").astype(str)
    return df.groupby("month").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="attendance_rate")


def build_agg_grade_course(grades, courses):
    df = grades.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    return df.groupby("course_name")["score"].mean().reset_index(name="avg_score").sort_values("avg_score")


def build_agg_kmeans_segments(attendance, grades, engagement, concepts):
    att11    = attendance.groupby("student_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    avg_g11  = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    logins11 = (
        engagement[engagement["event_type"] == "login"]
        .groupby("student_id").size().reset_index(name="logins")
    )
    video11  = (
        engagement[engagement["event_type"] == "video_watch"]
        .groupby("student_id")["duration_seconds"].sum().reset_index(name="video_sec")
    )
    failed11 = (
        concepts[concepts["mastery_status"] == "failed"]
        .groupby("student_id").size().reset_index(name="failed_c")
    )
    df11 = (
        att11
        .merge(avg_g11, on="student_id", how="outer")
        .merge(logins11, on="student_id", how="outer")
        .merge(video11[["student_id", "video_sec"]], on="student_id", how="outer")
        .merge(failed11, on="student_id", how="outer")
        .fillna(0)
    )
    features = ["att_rate", "avg_grade", "logins", "video_sec", "failed_c"]
    X = StandardScaler().fit_transform(df11[features])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df11["cluster"] = km.fit_predict(X)

    seg_stats = df11.groupby("cluster")[features].mean().reset_index()
    seg_labels: dict = {}
    for _, row in seg_stats.iterrows():
        c = row["cluster"]
        if row["avg_grade"] >= 75 and row["att_rate"] >= 75:
            seg_labels[c] = "🏆 High Achievers"
        elif row["avg_grade"] < 60 and row["att_rate"] < 50:
            seg_labels[c] = "🚨 At-Risk / Disengaged"
        elif row["logins"] >= df11["logins"].median() and row["avg_grade"] < 70:
            seg_labels[c] = "⚡ Engaged but Struggling"
        else:
            seg_labels[c] = "📈 Average / Improving"

    df11["segment"] = df11["cluster"].map(seg_labels)
    return df11


def build_agg_suggestions(concepts, attendance, groups, courses, students, grades, submissions):
    """Aggregate values used in the Suggestions tab."""
    fail_s = concepts.groupby("concept_name")["mastery_status"].apply(
        lambda x: (x == "failed").mean() * 100
    ).reset_index(name="fail_rate")
    worst_cs = fail_s.sort_values("fail_rate", ascending=False).iloc[0]["concept_name"]

    grp_att_s = attendance.groupby("group_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    grp_att_s = grp_att_s.merge(groups[["group_id", "group_name", "course_id"]], on="group_id", how="left")
    grp_att_s = grp_att_s.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    pav_s = grp_att_s["att_rate"].mean()
    low_att_s = grp_att_s[grp_att_s["att_rate"] < pav_s].sort_values("att_rate")

    late_rate_s = submissions["is_late"].mean() * 100
    true_sz_s   = students.groupby("group_id").size().reset_index(name="true_count")
    grp_sz_s    = groups[["group_id", "group_name"]].merge(true_sz_s, on="group_id", how="left").fillna(0)
    unviable_s  = grp_sz_s.sort_values("true_count").iloc[0]

    avg_grade_s = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    att_s = attendance.groupby("student_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    risk_s = (
        students[["student_id", "group_name", "course_name"]]
        .merge(avg_grade_s, on="student_id", how="left")
        .merge(att_s, on="student_id", how="left")
    )
    risk_s["risk_flag"] = (
        (risk_s["avg_grade"].fillna(0) < 55) | (risk_s["att_rate"].fillna(0) < 60)
    )
    risk_count_s = int(risk_s["risk_flag"].sum())
    lowest_att_group_s = (
        low_att_s.iloc[0] if len(low_att_s) else grp_att_s.sort_values("att_rate").iloc[0]
    )

    # Return as a single-row scalar table
    return pd.DataFrame([{
        "worst_concept":         worst_cs,
        "worst_concept_fail_pct": float(fail_s.sort_values("fail_rate", ascending=False).iloc[0]["fail_rate"]),
        "unviable_group_name":   str(unviable_s["group_name"]),
        "unviable_group_size":   int(unviable_s["true_count"]),
        "lowest_att_group_name": str(lowest_att_group_s["group_name"]),
        "lowest_att_group_rate": float(lowest_att_group_s["att_rate"]),
        "platform_avg_att":      float(pav_s),
        "risk_student_count":    risk_count_s,
        "late_rate_pct":         float(late_rate_s),
    }]), low_att_s


# Q-series (one per question key) 

def build_q_series(courses, groups, students, grades, attendance, concepts, engagement, submissions):
    """
    Build all Q1–Q15 pre-computed DataFrames.
    Returns a dict: { "q1_grp_att": df, "q2_stats2": df, ... }
    """
    series: dict[str, pd.DataFrame] = {}

    # Q1: Attendance rate per group
    grp_att = attendance.groupby("group_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    grp_att = grp_att.merge(groups[["group_id", "group_name", "course_id"]], on="group_id", how="left")
    grp_att = grp_att.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    grp_att = grp_att.sort_values("att_rate")
    series["q1_grp_att"] = grp_att

    att_m = attendance.copy()
    att_m["month"] = att_m["session_datetime"].dt.to_period("M").astype(str)
    monthly = att_m.groupby(["month", "group_id"]).apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="rate")
    monthly = monthly.merge(groups[["group_id", "group_name"]], on="group_id")
    series["q1_monthly"] = monthly


    # Q2: Score distribution by assessment type
    stats2 = grades.groupby("type")["score"].agg(["mean", "std", "median"]).reset_index()
    stats2.columns = ["Type", "Mean", "Std", "Median"]
    series["q2_stats"] = stats2


    # Q3: Course grade comparison
    g3 = grades.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    cstats = g3.groupby("course_name")["score"].agg(["mean", "std", "median", "count"]).reset_index()
    cstats.columns = ["Course", "Mean", "Std", "Median", "N"]
    cstats = cstats.sort_values("Mean", ascending=False)
    series["q3_cstats"] = cstats


    # Q4: Attendance vs grade
    att_s = attendance.groupby("student_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    avg_g = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    df4 = att_s.merge(avg_g, on="student_id").merge(
        students[["student_id", "course_name", "group_name"]], on="student_id", how="left"
    )
    df4["Band"] = pd.qcut(
        df4["att_rate"].rank(method="first"), 4,
        labels=["Q1 Low", "Q2", "Q3", "Q4 High"]
    ).astype(str)
    series["q4_df"] = df4
    qdf = df4.groupby("Band")["avg_grade"].mean().reset_index()
    series["q4_qdf"] = qdf

    # Q5: Engagement vs academic performance
    logins = engagement[engagement["event_type"] == "login"].groupby("student_id").size().reset_index(name="logins")
    video  = (
        engagement[engagement["event_type"] == "video_watch"]
        .groupby("student_id")["duration_seconds"].sum().reset_index(name="video_sec")
    )
    video["video_hrs"] = video["video_sec"] / 3600
    avg_g5 = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    df5 = (
        avg_g5
        .merge(logins, on="student_id", how="left")
        .merge(video[["student_id", "video_hrs"]], on="student_id", how="left")
        .merge(students[["student_id", "course_name"]], on="student_id", how="left")
    )
    df5[["logins", "video_hrs"]] = df5[["logins", "video_hrs"]].fillna(0)
    series["q5_df"] = df5
    eng_counts = engagement.groupby("event_type").size().reset_index(name="count")
    series["q5_eng_counts"] = eng_counts

    # Q6: Concept failure rates 
    fail6 = concepts.groupby(["concept_name", "course_id"])["mastery_status"].apply(
        lambda x: (x == "failed").mean() * 100
    ).reset_index(name="fail_rate")
    fail6 = fail6.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    fail6 = fail6.sort_values("fail_rate", ascending=False)
    series["q6_fail"] = fail6

    heat6 = concepts.groupby(["course_id", "concept_name"])["mastery_status"].apply(
        lambda x: (x == "failed").mean() * 100
    ).reset_index(name="fail_rate")
    heat6 = heat6.merge(courses[["course_id", "course_name"]], on="course_id")
    pivot6 = heat6.pivot_table(index="course_name", columns="concept_name", values="fail_rate", fill_value=0)
    # Store pivot as long format for MongoDB
    pivot6_long = pivot6.reset_index().melt(id_vars="course_name", var_name="concept_name", value_name="fail_rate")
    series["q6_pivot_long"] = pivot6_long

    # Q7: Weakest concept mastery over time 
    fail7 = concepts.groupby("concept_name")["mastery_status"].apply(
        lambda x: (x == "failed").mean() * 100
    ).reset_index(name="fail_rate")
    worst7 = fail7.sort_values("fail_rate", ascending=False).iloc[0]["concept_name"]
    wc = concepts[concepts["concept_name"] == worst7].copy()
    wc["month"] = wc["timestamp"].dt.to_period("M").astype(str)
    trend7 = wc.groupby("month").agg(
        pass_rate=("mastery_status", lambda x: (x == "passed").mean() * 100),
        attempts=("mastery_status", "count")
    ).reset_index()
    series["q7_trend"] = trend7
    series["q7_worst_name"] = pd.DataFrame([{"worst_concept": worst7}])

    # Q8: Late submissions vs score 
    sub8 = submissions.merge(
        grades[["assessment_id", "student_id", "score"]], on=["assessment_id", "student_id"], how="left"
    )
    sub8["buffer_hours"] = (sub8["deadline"] - sub8["submitted_at"]).dt.total_seconds() / 3600
    sub8 = sub8.dropna(subset=["score"])
    sub8["Band8"] = pd.cut(
        sub8["buffer_hours"].clip(-200, 200),
        bins=[-200, -1, 6, 24, 200],
        labels=["Late", "<6h", "6–24h", ">24h"]
    ).astype(str)
    # Drop datetime columns that are hard to serialize consistently
    sub8_store = sub8.drop(columns=["deadline", "submitted_at"], errors="ignore")
    series["q8_sub"] = sub8_store
    band_avg = sub8.groupby("Band8")["score"].mean().reset_index()
    series["q8_band_avg"] = band_avg

    # Q9: Attendance & engagement timeline
    att9 = attendance.copy()
    att9["week"] = att9["session_datetime"].dt.to_period("W").astype(str)
    att_wk = att9.groupby("week").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")

    eng9 = engagement.copy()
    eng9["week"] = eng9["event_datetime"].dt.to_period("W").astype(str)
    eng_wk = eng9.groupby("week").size().reset_index(name="events")

    merged9 = att_wk.merge(eng_wk, on="week", how="outer").sort_values("week")
    series["q9_merged"] = merged9

    att9["month"] = att9["session_datetime"].dt.month_name()
    att9["dow"]   = att9["session_datetime"].dt.day_name()
    heat9 = att9.groupby(["month", "dow"]).apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="rate")
    series["q9_heat"] = heat9

    # Q10: Age band analysis
    avg_g10  = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    att10    = attendance.groupby("student_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    logins10 = (
        engagement[engagement["event_type"] == "login"]
        .groupby("student_id").size().reset_index(name="logins")
    )
    df10 = (
        students[["student_id", "age"]]
        .merge(avg_g10, on="student_id", how="left")
        .merge(att10, on="student_id", how="left")
        .merge(logins10, on="student_id", how="left")
    )
    df10 = df10.dropna(subset=["age"])
    df10["age_band"] = pd.cut(
        df10["age"], bins=[0, 19, 22, 25, 30, 100],
        labels=["≤19", "20–22", "23–25", "26–30", "31+"]
    ).astype(str)
    band_stats = df10.groupby("age_band").agg(
        avg_grade=("avg_grade", "mean"),
        att_rate=("att_rate", "mean"),
        logins=("logins", "mean"),
        count=("student_id", "count")
    ).reset_index()
    series["q10_band_stats"] = band_stats

    # Q11: KMeans segmentation — built separately via build_agg_kmeans_segments ──

    # Q12: True vs stated group sizes 
    true_sz12 = students.groupby("group_id").size().reset_index(name="true_count")
    comp12 = groups[["group_id", "group_name", "course_id", "stated_num_students"]].merge(
        true_sz12, on="group_id", how="left"
    ).fillna(0)
    comp12 = comp12.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    comp12["discrepancy"] = comp12["stated_num_students"] - comp12["true_count"]
    comp12["disc_pct"] = (
        comp12["discrepancy"] / comp12["stated_num_students"].replace(0, np.nan) * 100
    ).fillna(100)
    comp12 = comp12.sort_values("true_count")
    series["q12_comp"] = comp12

    # Q13: Unviable group & merge recommendation 
    true_sz13 = students.groupby("group_id").size().reset_index(name="true_count")
    grp13 = groups[["group_id", "group_name", "course_id"]].merge(
        true_sz13, on="group_id", how="left"
    ).fillna(0)
    grp13 = grp13.merge(courses[["course_id", "course_name"]], on="course_id", how="left")
    series["q13_grp"] = grp13

    # Concept similarity for merge recommendation
    unviable = grp13.sort_values("true_count").iloc[0]
    if int(unviable["true_count"]) > 0:
        unv_sts = students[students["group_id"] == unviable["group_id"]]["student_id"].tolist()
        unv_con = concepts[concepts["student_id"].isin(unv_sts)].groupby("concept_name")["score_pct"].mean()
        other_grps = grp13[grp13["true_count"] > 10].copy()
        from scipy.stats import pearsonr
        sims = []
        for _, row in other_grps.iterrows():
            g_sts = students[students["group_id"] == row["group_id"]]["student_id"].tolist()
            g_con = concepts[concepts["student_id"].isin(g_sts)].groupby("concept_name")["score_pct"].mean()
            common = unv_con.index.intersection(g_con.index)
            if len(common) > 3:
                r_val, _ = pearsonr(unv_con[common], g_con[common])
                sims.append({
                    "group_name": row["group_name"],
                    "course_name": row["course_name"],
                    "true_count": int(row["true_count"]),
                    "similarity": r_val,
                })
        if sims:
            series["q13_sim"] = pd.DataFrame(sims).sort_values("similarity", ascending=False)

    # Q14: At-risk student ranking
    att14 = attendance.groupby("student_id").apply(
        lambda x: (x["status"] == "attended").mean() * 100
    ).reset_index(name="att_rate")
    eng14 = engagement.copy()
    eng14["month"] = eng14["event_datetime"].dt.to_period("M").astype(str)
    eng_trend = eng14.groupby(["student_id", "month"]).size().reset_index(name="events")
    months_sorted = sorted(eng_trend["month"].unique())
    mid = months_sorted[len(months_sorted) // 2]
    fh = eng_trend[eng_trend["month"] <= mid].groupby("student_id")["events"].mean().reset_index(name="eng_first")
    sh = eng_trend[eng_trend["month"] >  mid].groupby("student_id")["events"].mean().reset_index(name="eng_last")
    eng_delta = fh.merge(sh, on="student_id", how="outer").fillna(0)
    eng_delta["eng_decline"] = eng_delta["eng_first"] - eng_delta["eng_last"]
    avg_g14 = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    fail14  = (
        concepts[concepts["mastery_status"] == "failed"]
        .groupby("student_id").size().reset_index(name="failed_c")
    )
    df14 = (
        students[["student_id", "full_name", "course_name", "group_name"]]
        .merge(att14, on="student_id", how="left")
        .merge(eng_delta[["student_id", "eng_decline"]], on="student_id", how="left")
        .merge(avg_g14, on="student_id", how="left")
        .merge(fail14, on="student_id", how="left")
        .fillna(0)
    )

    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    df14["risk_score"] = (
        norm(100 - df14["att_rate"]) * 0.35
        + norm(df14["eng_decline"].clip(0)) * 0.25
        + norm(100 - df14["avg_grade"]) * 0.25
        + norm(df14["failed_c"]) * 0.15
    ) * 100
    series["q14_df"] = df14

    # Q15: Group grade trends
    g15 = grades.merge(groups[["group_id", "group_name"]], on="group_id", how="left").sort_values("date")
    g15["seq"] = g15.groupby("group_id")["date"].transform(
        lambda x: pd.factorize(x.astype(str))[0] + 1
    )
    trend15 = g15.groupby(["group_id", "group_name", "seq"])["score"].mean().reset_index()
    series["q15_trend"] = trend15

    from scipy.stats import linregress
    slopes15 = []
    for gid, gdf in trend15.groupby("group_id"):
        if len(gdf) >= 3:
            slope, _, r, _, _ = linregress(gdf["seq"], gdf["score"])
            slopes15.append({
                "group_name": gdf["group_name"].iloc[0],
                "slope": slope,
                "r": r,
            })
    slope_df = pd.DataFrame(slopes15).sort_values("slope", ascending=False)
    series["q15_slopes"] = slope_df

    fl = trend15.groupby("group_name").apply(
        lambda x: pd.Series({
            "First": x.sort_values("seq").iloc[0]["score"],
            "Last":  x.sort_values("seq").iloc[-1]["score"],
        })
    ).reset_index()
    fl["Change"] = fl["Last"] - fl["First"]
    series["q15_first_last"] = fl

    return series


# Main
def main():
    uri = get_uri()
    client = MongoClient(uri)
    db = client[DB_NAME]
    print(f"\n✅ Connected to MongoDB Atlas → database: '{DB_NAME}'")

    courses, groups, students, grades, attendance, concepts, engagement, submissions = load_raw()

    print("\n── Uploading raw collections ─────────────────────────────────────────")
    upload(db, "raw_courses",     courses)
    upload(db, "raw_groups",      groups)
    upload(db, "raw_students",    students)
    upload(db, "raw_grades",      grades)
    upload(db, "raw_attendance",  attendance)
    upload(db, "raw_concepts",    concepts)
    upload(db, "raw_engagement",  engagement)
    upload(db, "raw_submissions", submissions)

    print("\n── Building & uploading aggregated collections ───────────────────────")

    agg_student_risk = build_agg_student_risk(students, attendance, grades)
    upload(db, "agg_student_risk", agg_student_risk)

    agg_concept_fail = build_agg_concept_fail(concepts)
    upload(db, "agg_concept_fail", agg_concept_fail)

    agg_group_att = build_agg_group_att(attendance, groups, courses)
    upload(db, "agg_group_att", agg_group_att)

    agg_group_health = build_agg_group_health(students, groups)
    upload(db, "agg_group_health", agg_group_health)

    agg_monthly_att = build_agg_monthly_att(attendance)
    upload(db, "agg_monthly_att", agg_monthly_att)

    agg_grade_course = build_agg_grade_course(grades, courses)
    upload(db, "agg_grade_course", agg_grade_course)

    agg_kmeans = build_agg_kmeans_segments(attendance, grades, engagement, concepts)
    upload(db, "agg_kmeans_segments", agg_kmeans)

    agg_suggestions_meta, low_att_df = build_agg_suggestions(
        concepts, attendance, groups, courses, students, grades, submissions
    )
    upload(db, "agg_suggestions",         agg_suggestions_meta)
    upload(db, "agg_suggestions_low_att", low_att_df)

    print("\n── Building & uploading Q-series ────────────────────────────────────")
    q_series = build_q_series(
        courses, groups, students, grades, attendance, concepts, engagement, submissions
    )
    for key, df in q_series.items():
        upload(db, f"agg_q_{key}", df)

    client.close()
    print(f"\n🎉 Done — {DB_NAME} is ready. All collections uploaded to Atlas.")


if __name__ == "__main__":
    main()
