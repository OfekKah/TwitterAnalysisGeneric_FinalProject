import os
import tempfile
import pickle
import sqlite3
import pandas as pd
import pytest
from unittest import mock
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ייבוא הפונקציות שלך
from generic_code import (
    validate_db_file, read_authors, read_posts, analyze_group,
    plot_bar_chart,
    plot_tweet_distribution,
    plot_top_authors_by_tweet_count,
    plot_top_locations_by_count,
    plot_friends_count_range,
    plot_followers_count_range,
    plot_posts_count_range,
    plot_scatter_with_limits_multiple_groups,
    plot_tweets_distribution_per_month_merged
)
from generic_code import main as main_generic

from emotion_hate_sentiment_analysis import (
    load_data, process_tweets, clean_text
)
from emotion_hate_sentiment_analysis import main as main_emotion

# קבועים מדומים עבור הטסטים
REQUIRED_POSTS_COLUMNS = {
    "author",
    "date",
    "content"
}
OPTIONAL_POSTS_COLUMNS = {"label"}

REQUIRED_AUTHORS_COLUMNS = {
    "author_screen_name",
    "friends_count",
    "followers_count",
    "statuses_count",
    "location"
}

# Patch validate_columns בכל הטסטים כדי לא להתעסק בהגדרות טבלאות
@pytest.fixture(autouse=True)
def patch_validate_columns():
    with mock.patch("generic_code.validate_columns") as m:
        yield m

# --- טסטים על פונקציות בסיס נתונים ---

def test_validate_db_file_valid():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        validate_db_file(tmp.name)

def test_validate_db_file_not_found():
    with pytest.raises(FileNotFoundError):
        validate_db_file("non_existing_file.db")

def test_validate_db_file_wrong_extension():
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        with pytest.raises(ValueError):
            validate_db_file(tmp.name)

def test_read_authors(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE authors (
            author_screen_name TEXT,
            friends_count INTEGER,
            followers_count INTEGER,
            statuses_count INTEGER,
            location TEXT
        )
    """)
    conn.execute("""
        INSERT INTO authors VALUES ('user123', 50, 200, 100, 'TLV')
    """)
    conn.commit()

    df = read_authors(conn)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert "author_screen_name" in df.columns

# --- טסטים read_posts עם checkpoint ---

def test_read_posts_with_checkpoint(tmp_path):
    db_path = tmp_path / "test.db"
    checkpoint_file = tmp_path / "checkpoint.pkl"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE posts (
            author TEXT,
            date TEXT,
            content TEXT,
            label TEXT
        )
    """)
    conn.executemany("""
        INSERT INTO posts (author, date, content, label)
        VALUES (?, ?, ?, ?)""",
        [(f"user{i}", "2023-01-01", f"text {i}", None) for i in range(10)]
    )
    conn.commit()

    with open(checkpoint_file, "wb") as f:
        pickle.dump(5, f)

    chunks = list(read_posts(conn, chunksize=3, checkpoint_file=checkpoint_file))
    total_rows = sum(len(chunk) for chunk in chunks)
    assert total_rows == 5

# --- טסט load_data ---

def test_load_data(tmp_path):
    db_path = tmp_path / "test.db"
    checkpoint_file = tmp_path / "checkpoint.pkl"
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE authors (
            author_screen_name TEXT,
            friends_count INTEGER,
            followers_count INTEGER,
            statuses_count INTEGER,
            location TEXT
        )
    """)
    conn.execute("""
        INSERT INTO authors VALUES ('user_a', 150, 300, 1000, 'Jerusalem')
    """)

    conn.execute("""
        CREATE TABLE posts (
            author TEXT,
            date TEXT,
            content TEXT,
            label TEXT
        )
    """)
    conn.executemany("""
        INSERT INTO posts (author, date, content, label)
        VALUES (?, ?, ?, ?)""",
        [(f"user{i}", "2023-01-01", f"content {i}", None) for i in range(6)]
    )
    conn.commit()
    conn.close()

    posts_gen, authors_df = load_data(str(db_path), chunksize=2, checkpoint_file=str(checkpoint_file))
    assert authors_df.shape[0] == 1
    assert "author_screen_name" in authors_df.columns
    assert sum(len(chunk) for chunk in posts_gen) == 6

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(4, f)

    posts_gen2, _ = load_data(str(db_path), chunksize=2, checkpoint_file=str(checkpoint_file))
    assert sum(len(chunk) for chunk in posts_gen2) == 2

# --- טסטים לפונקציות טקסטים/רשתות חברתיות ---

def test_analyze_group_missing_column(capfd):
    test_df = pd.DataFrame({
        'followers_count': [100, 150, None],
        'statuses_count': [20, None, 10]
        # 'friends_count' intentionally missing
    })

    analyze_group(test_df, 'TestGroup', 'output', [0, 100, 1000], ['Low', 'Medium'], 'summary.csv')
    out, err = capfd.readouterr()
    assert "Column friends_count does not exist" in out

def test_clean_text_removal():
    raw = "Check this out ❤😊 http://example.com #hashtag @user"
    cleaned = clean_text(raw)

    # טקסט אחרי ניקוי צפוי לפי preprocess_tweet
    expected_keywords = ["check", "this", "out"]
    unexpected = ["http", "@", "#", "❤", "😊"]

    for word in expected_keywords:
        assert word in cleaned.lower(), f"Expected '{word}' to be in cleaned text"

    for token in unexpected:
        assert token not in cleaned, f"Unexpected token '{token}' found in cleaned text"

    # אפשר גם להוסיף אסרטיביות כללית:
    assert isinstance(cleaned, str)
    assert cleaned.strip() != ""

def test_process_tweets_creates_csv_and_checkpoint(tmp_path):
    # יצירת דאטה קטן לדוגמה
    df = pd.DataFrame({
        "author": ["user1", "user2"],
        "content": ["I love pizza!", "Sad day"],
        "new_date": ["2022-01-01", "2022-01-02"]
    })

    output_dir = tmp_path / "output"
    checkpoint_file = tmp_path / "checkpoint.pkl"

    # נשתמש ב־mock כדי להחליף את create_analyzer ולמנוע ריצת מודל אמיתי
    with mock.patch("emotion_hate_sentiment_analysis.create_analyzer") as mock_create_analyzer:
        mock_analyzer = mock.Mock()
        
        # נוודא שקריאת predict מחזירה תוצאה מדומה עם .probas
        mock_analyzer.predict.return_value = mock.Mock(probas={"joy": 0.9, "sadness": 0.1})
        mock_create_analyzer.return_value = mock_analyzer

        process_tweets(
            tweets_df=df,
            output_dir=output_dir,
            checkpoint_file=checkpoint_file,
            batch_size=1,
            max_batches=2,
            lang="en",
            task="emotion",
            num_files=1
        )

    # בדיקה שנשמר לפחות קובץ תוצאה אחד
    output_files = list(output_dir.glob("emotion_results_part*.csv"))
    assert output_files, "No CSV output file created"

    # בדיקה שהקובץ checkpoint נוצר
    assert checkpoint_file.exists(), "Checkpoint file not created"

    # בדיקה שתוכן הקובץ הגיוני
    result_df = pd.read_csv(output_files[0])
    assert "author" in result_df.columns
    assert "joy" in result_df.columns
    assert len(result_df) == 2
    
    
    
def test_all_plot_functions_create_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        # ---- 1. plot_bar_chart ----
        bar_data = pd.Series([10, 20, 15], index=['Low', 'Medium', 'High'])
        bar_path = os.path.join(tmpdir, "bar_chart.png")
        plot_bar_chart(bar_data, "Test Bar Chart", "Range", "Count", bar_path)
        assert os.path.isfile(bar_path) and os.path.getsize(bar_path) > 0

        # ---- 2. plot_tweet_distribution ----
        tweet_df = pd.DataFrame({
            'month': pd.period_range(start='2021-01', periods=3, freq='M').tolist()*2
        })
        tweet_path = os.path.join(tmpdir, "tweet_distribution.png")
        plot_tweet_distribution(tweet_df, "Test Tweet Distribution", tweet_path, tmpdir)
        assert os.path.isfile(tweet_path) and os.path.getsize(tweet_path) > 0

        # ---- 3. plot_top_authors_by_tweet_count ----
        author_df = pd.DataFrame({'author': ['a', 'b', 'a', 'c', 'a', 'b']})
        author_path = os.path.join(tmpdir, "group_top_authors.png")
        plot_top_authors_by_tweet_count(author_df, "group", tmpdir)
        assert os.path.isfile(author_path) and os.path.getsize(author_path) > 0

        # ---- 4. plot_top_locations_by_count ----
        location_df = pd.DataFrame({'country': ['US', 'US', 'IL', 'FR', 'FR', 'FR']})
        location_path = os.path.join(tmpdir, "group_top_countries.png")
        plot_top_locations_by_count(location_df, "group", tmpdir)
        assert os.path.isfile(location_path) and os.path.getsize(location_path) > 0

        # ---- 5-7. friends/followers/posts count ----
        fake_data = pd.DataFrame({
            'friends_count': [100, 400, 800],
            'followers_count': [300, 700, 1500],
            'statuses_count': [50, 300, 1000]
        })
        group_names = ['TestGroup']
        groups_data = [fake_data.copy()]
        for func, filename in zip(
            [plot_friends_count_range, plot_followers_count_range, plot_posts_count_range],
            ["Friends_Count_Range_Merged.png", "Followers_Count_Range_Merged.png", "Posts_Count_Range_Merged.png"]
        ):
            path = os.path.join(tmpdir, filename)
            func(groups_data, group_names, tmpdir, filename)
            assert os.path.isfile(path) and os.path.getsize(path) > 0

        # ---- 8. scatter plot ----
        scatter_path = os.path.join(tmpdir, "Scatter_Test.png")
        plot_scatter_with_limits_multiple_groups(
            data_groups=[fake_data],
            labels=["TestGroup"],
            x_column="followers_count",
            y_column="statuses_count",
            x_limit=2000,
            y_limit=2000,
            title="Scatter Test",
            xlabel="Followers",
            ylabel="Posts",
            output_folder=tmpdir
        )
        assert os.path.isfile(scatter_path) and os.path.getsize(scatter_path) > 0

        # ---- 9. tweets distribution merged ----
        tweet_df2 = pd.DataFrame({
            'date': ['2022-01-01', '2022-02-01', '2022-02-15']
        })
        merged_path = os.path.join(tmpdir, "Tweet_Distribution_by_Month_for_All_Groups.png")
        plot_tweets_distribution_per_month_merged([tweet_df2], ["GroupA"], "Tweet Distribution by Month for All Groups", tmpdir)
        assert os.path.isfile(merged_path) and os.path.getsize(merged_path) > 0

def create_test_db_with_labels(db_path):
    conn = sqlite3.connect(db_path)

    # authors table
    conn.execute("""
        CREATE TABLE authors (
            author_screen_name TEXT,
            friends_count INTEGER,
            followers_count INTEGER,
            statuses_count INTEGER,
            location TEXT,
            label TEXT
        )
    """)
    conn.execute("""
        INSERT INTO authors VALUES 
        ('user1', 100, 500, 1000, 'Tel Aviv', 'GroupA'),
        ('user2', 200, 700, 300, 'Jerusalem', 'GroupB')
    """)

    # posts table
    conn.execute("""
        CREATE TABLE posts (
            author TEXT,
            date TEXT,
            content TEXT
        )
    """)
    conn.executemany("""
        INSERT INTO posts VALUES (?, ?, ?)
    """, [
        ('user1', '2022-01-01 12:00:00', 'I love pizza'),
        ('user2', '2022-02-01 13:30:00', 'Terrible news today'),
    ])

    conn.commit()
    conn.close()

def test_main_generic_pipeline_runs_without_errors(tmp_path, capsys):
    db_path = tmp_path / "test_generic.db"
    create_test_db_with_labels(db_path)

    with tempfile.TemporaryDirectory() as temp_output:
        os.chdir(temp_output)

        main_generic(str(db_path))

        output_folder = Path(temp_output) / "output"
        assert output_folder.exists(), "תיקיית פלט לא נוצרה"

        output_files = list(output_folder.glob("*.csv")) + list(output_folder.glob("*.png"))
        assert output_files, "לא נוצרו קבצי פלט (csv או גרפים)"

        out, err = capsys.readouterr()
        assert "An error occurred:" not in out

def test_main_emotion_pipeline_runs_without_errors(tmp_path, capsys):
    db_path = tmp_path / "test_emotion.db"
    create_test_db_with_labels(db_path)

    save_dir = tmp_path / "output_analysis2"

    # ניצור את כל תיקיות ה-checkpoint שהקוד ישתמש בהן
    for task in ["emotion", "sentiment", "hate_speech"]:
        (save_dir / f"{task}_results").mkdir(parents=True, exist_ok=True)
# 
    checkpoint_path = tmp_path / "posts_checkpoint.pkl"
# 
    main_emotion(
        str(db_path),
        save_dir=str(save_dir),
        batch_size=10,
        max_batches=1,
        num_files=1,
        # 
        checkpoint_file=checkpoint_path
        # 
    )

    assert save_dir.exists(), "תיקיית פלט לא נוצרה"

    for task in ["emotion", "sentiment", "hate_speech"]:
        task_dir = save_dir / f"{task}_results"
        assert task_dir.exists(), f"תיקיית {task} לא נוצרה"
        csv_files = list(task_dir.glob("*.csv"))
        assert csv_files, f"לא נוצר קובץ תוצאה עבור {task}"

        graph_dir = save_dir / f"{task}_analysis_graphs"
        assert graph_dir.exists(), f"תיקיית גרפים {task}_analysis_graphs לא קיימת"

    out, err = capsys.readouterr()
    assert "error" not in out.lower()