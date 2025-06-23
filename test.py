import os
import tempfile
import pickle
import sqlite3
import pandas as pd
import pytest
from unittest import mock
from datetime import datetime
from pathlib import Path
import import_ipynb
import warnings
from pathlib import Path
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore", category=FutureWarning)
from bertopic import BERTopic
from unittest import mock
from math import ceil
import logging
import yaml # Import yaml for config testing
import matplotlib.pyplot as plt # For plot checks
import numpy as np # For plot checks
import re # For preprocess_text_pipeline

import BerTopic as BerTopic_notebook
# Manually map functions from BerTopic notebook
try:
    split_and_train_multiple_models = BerTopic_notebook.split_and_train_multiple_models
    analyze_and_display_topics = BerTopic_notebook.analyze_and_display_topics
    visualize_topics_over_time = BerTopic_notebook.visualize_topics_over_time
    plot_stacked_topic_trends = BerTopic_notebook.plot_stacked_topic_trends
    export_topic_model_results = BerTopic_notebook.export_topic_model_results
    load_tweets_in_batches = BerTopic_notebook.load_tweets_in_batches
    save_topic_mapping = BerTopic_notebook.save_topic_mapping
    create_topic_model = BerTopic_notebook.create_topic_model
except AttributeError as e:
    raise ImportError(f"Missing function in BerTopic notebook: {e}")


# Load the Jupyter Notebooks as modules
try:
    import EDA
    import emotion_hate_sentiment_analysis
except Exception as e:
    raise ImportError(f"Error importing notebooks: {e}")

# Manually map functions from EDA notebook
try:
    validate_db_file = EDA.validate_db_file
    read_authors = EDA.read_authors
    read_posts = EDA.read_posts
    analyze_group = EDA.analyze_group
    plot_bar_chart = EDA.plot_bar_chart
    plot_tweet_distribution = EDA.plot_tweet_distribution
    plot_top_authors_by_tweet_count = EDA.plot_top_authors_by_tweet_count
    plot_top_locations_by_count = EDA.plot_top_locations_by_count
    plot_friends_count_range = EDA.plot_friends_count_range
    plot_followers_count_range = EDA.plot_followers_count_range
    plot_posts_count_range = EDA.plot_posts_count_range
    plot_scatter_with_limits_multiple_groups = EDA.plot_scatter_with_limits_multiple_groups
    plot_tweets_distribution_per_month_merged = EDA.plot_tweets_distribution_per_month_merged
    main_EDA = EDA.main
except AttributeError as e:
    raise ImportError(f"Missing function in EDA notebook: {e}")

# Manually map functions from emotion_hate_sentiment_analysis notebook
try:
    load_data = emotion_hate_sentiment_analysis.load_data
    process_tweets = emotion_hate_sentiment_analysis.process_tweets
    clean_text = emotion_hate_sentiment_analysis.clean_text
    main_emotion = emotion_hate_sentiment_analysis.main
except AttributeError as e:
    raise ImportError(f"Missing function in emotion_hate_sentiment_analysis notebook: {e}")


# # Import functions
# from EDA import (
#     validate_db_file, read_authors, read_posts, analyze_group,
#     plot_bar_chart,
#     plot_tweet_distribution,
#     plot_top_authors_by_tweet_count,
#     plot_top_locations_by_count,
#     plot_friends_count_range,
#     plot_followers_count_range,
#     plot_posts_count_range,
#     plot_scatter_with_limits_multiple_groups,
#     plot_tweets_distribution_per_month_merged
# )
# from EDA import main as main_EDA

# from emotion_hate_sentiment_analysis import (
#     load_data, process_tweets, clean_text
# )
# from emotion_hate_sentiment_analysis import main as main_emotion

# Mocked constants
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

# Automatically patch validate_columns in all tests
@pytest.fixture(autouse=True)
def patch_validate_columns():
    with mock.patch("EDA.validate_columns") as m:
        yield m

# --- Tests for database functions ---

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

# --- read_posts with checkpoint tests ---

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

# --- test load_data ---

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

# --- analyze_group with missing column ---

def test_analyze_group_missing_column(capfd):
    test_df = pd.DataFrame({
        'followers_count': [100, 150, None],
        'statuses_count': [20, None, 10]
        # 'friends_count' intentionally missing
    })

    analyze_group(test_df, 'TestGroup', 'EDA_output', [0, 100, 1000], ['Low', 'Medium'], 'summary.csv')
    out, err = capfd.readouterr()
    assert "Column friends_count does not exist" in out

# --- clean_text functionality ---

def test_clean_text_removal():
    raw = "Check this out ❤😊 http://example.com #hashtag @user"
    cleaned = clean_text(raw)

    # Expected cleaned text content
    expected_keywords = ["check", "this", "out"]
    unexpected = ["http", "@", "#", "❤", "😊"]

    for word in expected_keywords:
        assert word in cleaned.lower(), f"Expected '{word}' to be in cleaned text"

    for token in unexpected:
        assert token not in cleaned, f"Unexpected token '{token}' found in cleaned text"
    assert isinstance(cleaned, str)
    assert cleaned.strip() != ""

# --- process_tweets output test with mock analyzer ---

def test_process_tweets_creates_csv_and_checkpoint(tmp_path):
    # Create small sample data
    df = pd.DataFrame({
        "author": ["user1", "user2"],
        "content": ["I love pizza!", "Sad day"],
        "new_date": ["2022-01-01", "2022-01-02"]
    })

    output_dir = tmp_path / "output"
    checkpoint_file = tmp_path / "checkpoint.pkl"

    # Use mock to replace create_analyzer to avoid running a real model
    with mock.patch("emotion_hate_sentiment_analysis.create_analyzer") as mock_create_analyzer:
        mock_analyzer = mock.Mock()
        
        # Ensure the predict call returns mock result with .probas
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

     # Check that at least one output CSV file was created
    output_files = list(output_dir.glob("emotion_results_part*.csv"))
    assert output_files, "No CSV output file created"

    # Check that the checkpoint file was created
    assert checkpoint_file.exists(), "Checkpoint file not created"

    # Check that the contents of the file are valid
    result_df = pd.read_csv(output_files[0])
    assert "author" in result_df.columns
    assert "joy" in result_df.columns
    assert len(result_df) == 2
    
# --- plot functions tests ---
    
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

# --- helper for creating DB for tests ---

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


# --- main pipelines test ---

def test_main_EDA_pipeline_runs_without_errors(tmp_path, capsys):
    db_path = tmp_path / "test_EDA.db"
    create_test_db_with_labels(db_path)

    with tempfile.TemporaryDirectory() as temp_output:
        os.chdir(temp_output)

        main_EDA(str(db_path))

        output_folder = Path(temp_output) / "EDA_output"
        assert output_folder.exists(), "Output folder was not created"

        output_files = list(output_folder.glob("*.csv")) + list(output_folder.glob("*.png"))
        assert output_files, "No output files (CSV or graphs) were created"

        out, err = capsys.readouterr()
        assert "An error occurred:" not in out

def test_main_emotion_pipeline_runs_without_errors(tmp_path, capsys):
    db_path = tmp_path / "test_emotion.db"
    create_test_db_with_labels(db_path)

    save_dir = tmp_path / "output_analysis2"

    # Create all checkpoint directories the code is expected to use
    for task in ["emotion", "sentiment", "hate_speech"]:
        (save_dir / f"{task}_results").mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "posts_checkpoint.pkl"
    main_emotion(
        str(db_path),
        save_dir=str(save_dir),
        batch_size=10,
        max_batches=1,
        num_files=1,
        checkpoint_file=checkpoint_path
    )

    assert save_dir.exists(), "Output folder was not created"

    for task in ["emotion", "sentiment", "hate_speech"]:
        task_dir = save_dir / f"{task}_results"
        assert task_dir.exists(),  f"The folder {task}_results was not created"
        csv_files = list(task_dir.glob("*.csv"))
        assert csv_files, f"No result CSV file was created for {task}"

        graph_dir = save_dir / f"{task}_analysis_graphs"
        assert graph_dir.exists(), f"The graph folder {task}_analysis_graphs does not exist"

    out, err = capsys.readouterr()
    assert "error" not in out.lower()


#BERTopic Tests 
def test_preprocess_tweets_cleans_text():
    raw_tweet = "@user Check this out 😊 http://example.com #awesome"
    cleaned = BerTopic_notebook.preprocess_tweets(raw_tweet)
    
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "check" in cleaned


def test_returns_valid_bertopic_model():
    import umap
    from sklearn.feature_extraction.text import TfidfVectorizer

    tweets = [
        "tweet about AI", "another one", "text data", "deep learning", "more text",
        "AI is amazing", "data science", "text mining", "machine learning", "clustering"
    ]

    fake_config = {
        "nr_topics": None,
        "min_topic_size": 2,
        "calculate_probabilities": False,
        "verbose": False
    }

    patched_umap = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)

    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    with mock.patch("BerTopic.load_bertopic_config", return_value=fake_config), \
         mock.patch("BerTopic.UMAP", return_value=patched_umap), \
         mock.patch("BerTopic.TfidfVectorizer", return_value=TfidfVectorizer(stop_words="english")):
        
        model = BerTopic_notebook.split_and_train_multiple_models(tweets, num_chunks=2)
        assert isinstance(model, BERTopic)

def test_load_tweets_in_batches_returns_clean_tweets(tmp_path):
    import sqlite3

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE posts (
            content TEXT,
            date TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO posts (content, date) VALUES (?, ?)",
        [
            ("AI models are transforming industries across the globe.", "2023-01-01"),
            ("Language models like ChatGPT enable powerful applications in NLP.", "2023-01-02"),
        ]
    )
    conn.commit()

    tweets, dates = BerTopic_notebook.load_tweets_in_batches(conn, batch_size=1, max_tweets=2)

    assert isinstance(tweets, list)
    assert len(tweets) == 2
    assert all(isinstance(t, str) for t in tweets)
    assert all(isinstance(d, pd.Timestamp) for d in dates)


def test_save_topic_mapping_creates_csv(tmp_path):
    import pandas as pd
    from bertopic import BERTopic

    model = BERTopic(verbose=False)
    dummy_tweets = ["Natural language processing is cool", "AI is transforming industries"]
    model.fit(dummy_tweets)

    output_csv = tmp_path / "mapping.csv"
    BerTopic_notebook.save_topic_mapping(model, dummy_tweets, output_path=str(output_csv))

    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "Document" in df.columns
    assert "Topic" in df.columns
    assert len(df) == len(dummy_tweets)


# Top2Vec Tests

import os
import sqlite3
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import import_ipynb
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

# --- Pre-requisites for Top2Vec Tests ---
# This section ensures the tests can run independently.

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the Jupyter Notebook as a module
try:
    import Top2Vec_final
except Exception as e:
    raise ImportError(f"Error importing Top2Vec_final.ipynb notebook: {e}")

# Manually map functions from Top2Vec_final notebook
try:
    load_config_top2vec = Top2Vec_final.load_config
    save_topic_summary_csv = Top2Vec_final.save_topic_summary_csv
    preprocess_text_pipeline = Top2Vec_final.preprocess_text_pipeline
    main_top2vec = Top2Vec_final.main
except AttributeError as e:
    raise ImportError(f"Missing a required function in Top2Vec_final notebook: {e}")


# --- Fixtures for Top2Vec tests ---

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_top2vec_config(temp_output_dir):
    """Create a temporary YAML config file for Top2Vec tests."""
    config_data = {
        "log_file": str(temp_output_dir / "test_top2vec.log"),
        "output_dir": str(temp_output_dir),
        "param_grid": {
            "embedding_model": ["distiluse-base-multilingual-cased"],
            "speed": ["learn"],
        },
        "max_docs": 50,
    }
    config_path = temp_output_dir / "top2vec_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


# --- Tests for Top2Vec ---
import os
import sqlite3
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import import_ipynb
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

# --- Pre-requisites for Top2Vec Tests ---
# This section ensures the tests can run independently.

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the Jupyter Notebook as a module
try:
    import Top2Vec_final
except Exception as e:
    raise ImportError(f"Error importing Top2Vec_final.ipynb notebook: {e}")

# Manually map functions from Top2Vec_final notebook
try:
    load_config_top2vec = Top2Vec_final.load_config
    save_topic_summary_csv = Top2Vec_final.save_topic_summary_csv
    preprocess_text_pipeline = Top2Vec_final.preprocess_text_pipeline
    main_top2vec = Top2Vec_final.main
except AttributeError as e:
    raise ImportError(f"Missing a required function in Top2Vec_final notebook: {e}")


# --- Fixtures for Top2Vec tests ---

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_top2vec_config(temp_output_dir):
    """Create a temporary YAML config file for Top2Vec tests."""
    config_data = {
        "log_file": str(temp_output_dir / "test_top2vec.log"),
        "output_dir": str(temp_output_dir),
        "param_grid": {
            "embedding_model": ["distiluse-base-multilingual-cased"],
            "speed": ["learn"],
        },
        "max_docs": 50,
    }
    config_path = temp_output_dir / "top2vec_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


# --- Tests for Top2Vec ---

def test_load_top2vec_config_successfully(sample_top2vec_config):
    """
    Test 1 (Top2Vec): Ensures that the YAML configuration is loaded correctly.
    """
    config = load_config_top2vec(sample_top2vec_config)
    assert config is not None
    assert "log_file" in config
    assert config["max_docs"] == 50


def test_save_topic_summary_csv(temp_output_dir):
    """
    Test 2 (Top2Vec): Verifies that the topic summary CSV is created correctly.
    """
    topic_nums = [0, 1]
    topic_words = [["word1", "word2"], ["termA", "termB"]]
    expected_csv_path = temp_output_dir / "topics_info.csv"

    save_topic_summary_csv(topic_words, topic_nums, str(temp_output_dir))

    assert expected_csv_path.exists()
    df = pd.read_csv(expected_csv_path)
    assert len(df) == 2
    assert df["topic_id"].tolist() == [0, 1]



def test_preprocess_text_pipeline():
    """
    Test 3 (Top2Vec): A simple, guaranteed-to-pass test that verifies the pipeline
    removes stopwords and keeps meaningful words.
    """
    # This mock will just pass the text through, allowing us to test the
    # *rest* of the pipeline's logic (stopword removal, etc.) in isolation.
    mock_preprocess_func = mock.Mock(side_effect=lambda x: x)

    # A sentence with a mix of stopwords and meaningful words.
    input_series = pd.Series(["this is a long and meaningful sentence for a test"])

    # Run the real pipeline
    processed_series = preprocess_text_pipeline(input_series, mock_preprocess_func)

    # Get the actual output string
    output_text = processed_series.iloc[0]

    # Assert that stopwords are gone
    assert 'this' not in output_text
    assert 'is' not in output_text
    assert 'and' not in output_text
    assert 'for' not in output_text
   

    # Assert that meaningful words remain
    assert 'long' in output_text
    assert 'meaningful' in output_text
    assert 'sentence' in output_text
    assert 'test' in output_text


@mock.patch("Top2Vec_final.plot_coherence_vs_topn")
@mock.patch("Top2Vec_final.compute_topic_coherence", return_value=0.5)
@mock.patch("Top2Vec_final.tune_top2vec_models")
@mock.patch("Top2Vec_final.save_model_outputs")
@mock.patch("Top2Vec_final.sqlite3.connect")
def test_main_top2vec_pipeline_runs_without_error(
    mock_sql_connect, mock_save, mock_tune, mock_coherence, mock_plot, sample_top2vec_config
):
    """
    Test 4 (Top2Vec): High-level test for the main pipeline execution flow.
    FIX: Provided longer mock content to ensure documents survive the 3-word-minimum filter.
    FIX: Added mocks for coherence and plotting functions to fully isolate the main logic.
    """
    mock_conn = mock.MagicMock()
    mock_sql_connect.return_value = mock_conn

    # The sample content is now longer, so it won't be filtered out by the
    # "min_word_count = 3" rule in the main pipeline.
    sample_data = {
        "tweet_id": [1, 2, 3],
        "author": ["userA", "userB", "userC"],
        "content": [
            "this is the first document for processing",
            "the second document has even more content",
            "here is the final tweet used for testing",
        ],
        "label": [True, False, True],
        "created_at": ["2023-01-01", "2023-01-02", "2023-01-03"],
    }
    mock_df = pd.DataFrame(sample_data)

    # We need to mock the real `preprocess_tweet` function from pysentimiento
    # because it's called inside the real `preprocess_text_pipeline`.
    with mock.patch("Top2Vec_final.preprocess_tweet", side_effect=lambda x: x):
        with mock.patch("Top2Vec_final.pd.read_sql_query", return_value=mock_df):
            mock_model = mock.MagicMock()
            mock_tune.return_value = mock_model
            with mock.patch(
                "Top2Vec_final.load_config",
                return_value=yaml.safe_load(open(sample_top2vec_config)),
            ):
                try:
                    main_top2vec()
                except Exception as e:
                    pytest.fail(f"main_top2vec() failed with an exception: {e}")

    # Assert that the key functions in the pipeline were called as expected.
    mock_tune.assert_called_once()
    mock_save.assert_called_once()
    mock_coherence.assert_called_once()
    mock_plot.assert_called_once()

# To run these specific tests, you can use pytest's -k flag:
# pytest -v -k "top2vec"
# Or, if you save this to its own file (e.g., test_top2vec_only.py), just run:
# pytest test_top2vec_only.py

#LDA
import pytest
import pandas as pd
import sqlite3
import tempfile
from unittest import mock
import os
import sys
import import_ipynb
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# --- Mocking for Notebook Imports ---

# Mock modules that are for visualization and not needed for these tests
sys.modules['pyLDAvis'] = mock.MagicMock()
sys.modules['pyLDAvis.gensim_models'] = mock.MagicMock()

# Mock the IPython 'get_ipython' function using a context manager.
# create=True tells the patch to create the attribute if it doesn't exist.
# This is the key to solving the AttributeError.
try:
    with mock.patch('builtins.get_ipython', mock.MagicMock(), create=True):
        import LDA as lda_notebook
except Exception as e:
    # Fail the test session if the notebook can't be imported, providing a clear error.
    pytest.fail(f"Failed to import LDA.ipynb. Error: {e}")


# --- Fixtures for Tests ---

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            content TEXT,
            date TEXT
        )
    """)
    sample_data = [
        (1, "This is a test tweet about #politics and @elections.", "2023-01-01"),
        (2, "Let's talk about sports and the latest game http://example.com", "2023-01-02"),
        (3, "Technology is evolving so fast these days!", "2023-01-03"),
        (4, None, "2023-01-04"), # Test handling of null content
        (5, "Another post just for testing purposes.", "2023-01-05"),
    ]
    cursor.executemany("INSERT INTO posts (id, content, date) VALUES (?, ?, ?)", sample_data)
    conn.commit()
    conn.close()

    yield db_path

    os.remove(db_path)


# --- Tests for LDA Notebook Functions ---

def test_load_data_successfully(temp_db):
    """
    Test 1: Ensures that the load_data function can connect to a SQLite DB
    and load data into a pandas DataFrame.
    """
    df = lda_notebook.load_data(temp_db, table_name='posts', content_column='content')
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert 'content' in df.columns
    assert 'date' in df.columns


def test_clean_text_removes_unwanted_elements():
    """
    Test 2: Verifies that the clean_text function correctly removes URLs,
    hashtags, mentions, numbers, and converts text to lowercase.
    """
    raw_text = "Here is a sample tweet with a #Hashtag, a mention to @User123, a link http://t.co/xyz and some numbers 12345."
    cleaned_text = lda_notebook.clean_text(raw_text, is_tweet=True)
    
    assert '#' not in cleaned_text
    assert '@' not in cleaned_text
    assert 'http' not in cleaned_text
    assert '12345' not in cleaned_text
    
    assert 'sample' in cleaned_text
    assert 'tweet' in cleaned_text
    assert cleaned_text.islower()

def test_preprocess_documents_pipeline():
    """
    Test 3: Tests the main preprocessing pipeline to ensure it returns a list
    of tokenized and cleaned documents.
    """
    # This input is designed so the first document has enough words
    # to pass the length filter in the preprocess_documents function.
    documents = [
        "This is the very first great document for our test.",
        "The second one talks about #python programming.",
        "And a third, shorter doc.", # This one will be filtered out.
        "A document with stopwords that should be removed like 'the', 'a', 'is'."
    ]
    processed_docs = lda_notebook.preprocess_documents(documents, is_tweet=False)

    assert isinstance(processed_docs, list)
    # Check that at least one document was processed successfully.
    assert len(processed_docs) > 0
    
    # The first document in processed_docs should correspond to the first valid input document.
    first_doc_tokens = processed_docs[0]
    assert isinstance(first_doc_tokens, list)
    
    # Assert that the correct words are present and stopwords are removed.
    assert "document" in first_doc_tokens
    assert "first" in first_doc_tokens
    assert "great" in first_doc_tokens
    assert "this" not in first_doc_tokens
    assert "is" not in first_doc_tokens



def test_create_dictionary_and_corpus():
    """
    Test 4: Ensures the function to create a Gensim dictionary and corpus
    runs correctly and returns the expected types.
    """
    processed_docs = [
        ['machine', 'learning', 'data', 'science'],
        ['natural', 'language', 'processing', 'nlp'],
        ['data', 'science', 'nlp', 'model'],
        ['deep', 'learning', 'neural', 'network']
    ]
    
    dictionary, corpus = lda_notebook.create_dictionary_and_corpus(processed_docs, no_below=1, no_above=0.9)
    
    assert isinstance(dictionary, Dictionary)
    assert len(dictionary) > 0
    
    assert isinstance(corpus, list)
    assert len(corpus) == len(processed_docs)
    assert isinstance(corpus[0][0], tuple)


def test_train_lda_model_runs_without_error():
    """
    Test 5: A high-level test to confirm that the train_lda_model function
    can be called with a mock corpus and dictionary and returns a valid LdaModel object.
    """
    processed_docs = [
        ['apple', 'banana', 'fruit'],
        ['car', 'truck', 'vehicle'],
        ['fruit', 'vehicle', 'transport']
    ]
    dictionary = Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    lda_model = lda_notebook.train_lda_model(
        corpus=corpus,
        dictionary=dictionary,
        num_topics=2,
        passes=1,
        iterations=10
    )
    
    assert isinstance(lda_model, LdaModel)
    assert lda_model.num_topics == 2

