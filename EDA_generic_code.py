import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import os
import subprocess
import sys

def read_db(conn, table):
    """Retrieve all rows from a database table."""
    query = f"SELECT * FROM {table}"
    return pd.read_sql(query, conn)

def describe_column(df, column):
    """Provide statistical description of a specified column."""
    return df[column].describe()

def categorize_counts(df, column, bins, labels):
    """Categorize a column into ranges and return value counts."""
    df = df.copy()  # Ensure we're working on a copy
    df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')  
    df = df.dropna(subset=[column])
    return pd.cut(df[column], bins=bins, labels=labels).value_counts().sort_index()

def plot_bar_chart(data, title, xlabel, ylabel, filename):
    """Plot a bar chart and save it to a file."""
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(filename)
    plt.close()

def process_tweets(df, column='date'):
    """Preprocess tweets to extract date and month."""
    df = df.copy()  # Create a copy to avoid warnings
    df.loc[:, 'date_new'] = pd.to_datetime(df[column].apply(lambda d: d.split()[0]), errors='coerce')
    df.loc[:, 'month'] = df['date_new'].dt.to_period('M')
    return df

def plot_tweet_distribution(df, title, filename, output_folder):
    if os.path.commonpath([output_folder, filename]) == output_folder:
        filename = os.path.relpath(filename, output_folder)

    tweets_per_month = df['month'].value_counts().sort_index()
    plt.figure(figsize=(20, 8))
    sns.barplot(x=tweets_per_month.index.astype(str), y=tweets_per_month.values, palette="viridis")
    plt.title(title)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    plt.xticks(rotation=65, fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_folder, filename)
    print(f"Saving graph to: {save_path}")
    plt.savefig(save_path)
    plt.close()

def analyze_group(data, group_name, output_folder, bins, labels, summary_file):
    """Analyze a single group and generate visualizations."""
    print(f"Analyzing group: {group_name}")
    print(f"Data size: {data.shape}")

    all_descriptions = []

    for column in ['friends_count', 'followers_count', 'statuses_count']:
        if column in data.columns:
            print(f"Missing values in {column}: {data[column].isnull().sum()}")
            description = describe_column(data, column).to_frame().T
            description['group'] = group_name
            description['column'] = column
            all_descriptions.append(description)
        else:
            print(f"Column {column} does not exist in the data.")

    if all_descriptions:
        combined_descriptions = pd.concat(all_descriptions, ignore_index=True)
        combined_descriptions = combined_descriptions.iloc[:,::-1]
        if not os.path.exists(summary_file):
            combined_descriptions.to_csv(summary_file, index=False)
        else:
            combined_descriptions.to_csv(summary_file, mode='a', index=False, header=False)
        print(f"Appended statistics for {group_name} to {summary_file}")

    for col, title in zip(['friends_count', 'followers_count', 'statuses_count'],
                          ["Friends Count", "Followers Count", "Posts Count"]):
        if col in data.columns:
            categorized = categorize_counts(data, col, bins, labels)
            chart_path = os.path.join(output_folder, f"{group_name}_{col}_range.png")
            plot_bar_chart(categorized, f"{group_name} by {title} Range", "Count Range", "Number of Individuals", chart_path)

def analyze_groups(groups_data, group_names, output_folder, bins, labels, summary_file):
    """Analyze multiple groups and generate visualizations."""
    for df, group_name in zip(groups_data, group_names):
        analyze_group(df, group_name, output_folder, bins, labels, summary_file)

def plot_friends_count_range(groups_data, group_names, output_folder, filename):
    # Define the bins and labels
    bins = [0, 50, 100, 200, 400, 700, 1000, 1500, float('inf')]
    labels = ['<50', '50-100', '100-200', '200-400', '400-700', '700-1000', '1000-1500', '1500>']
    # Initialize the dictionary
    profession_counts = {}
    for df, group_name in zip(groups_data, group_names):
        # Convert 'friends_count' to numeric, coercing errors to NaN
        df = df.copy()  # Ensure we're working on a copy
        df.loc[:, 'friends_count'] = pd.to_numeric(df['friends_count'], errors='coerce')  
        # Drop rows with NaN values in 'friends_count'
        df = df.dropna(subset=['friends_count'])
        # Categorize friends count using consistent binning
        df['friends_count_range'] = pd.cut(df['friends_count'], bins=bins, labels=labels)
        # Count the number in each range
        df_range_counts = df['friends_count_range'].value_counts().sort_index()
        profession_counts[group_name] = df_range_counts

    # Combine the counts into a DataFrame
    combined_counts = pd.DataFrame(profession_counts)

    # Plot the combined data
    combined_counts.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
    plt.title('Number of Friends Count Range for each populations')
    plt.xlabel('Friends Count Range')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=0)
    plt.legend(title='Group')
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path)
    plt.close()

def plot_followers_count_range(groups_data, group_names, output_folder, filename):
    # Define the bins and labels
    bins = [0, 50, 100, 200, 400, 700, 1000, 1500, float('inf')]
    labels = ['<50', '50-100', '100-200', '200-400', '400-700', '700-1000', '1000-1500', '1500>']
    # Initialize the dictionary
    profession_counts = {}
    for df, group_name in zip(groups_data, group_names):
        # Convert 'followers_count' to numeric, coercing errors to NaN
        df = df.copy()  # Ensure we're working on a copy
        df.loc[:, 'followers_count'] = pd.to_numeric(df['followers_count'], errors='coerce')  
        # Drop rows with NaN values in 'followers_count'
        df = df.dropna(subset=['followers_count'])
        # Categorize friends count using consistent binning
        df['followers_count_range'] = pd.cut(df['followers_count'], bins=bins, labels=labels)
        # Count the number in each range
        df_range_counts = df['followers_count_range'].value_counts().sort_index()
        profession_counts[group_name] = df_range_counts

    # Combine the counts into a DataFrame
    combined_counts = pd.DataFrame(profession_counts)

    # Plot the combined data
    combined_counts.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
    plt.title('Number of followers Count Range for each populations')
    plt.xlabel('followers Count Range')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=0)
    plt.legend(title='Group')
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path)
    plt.close()

def plot_posts_count_range(groups_data, group_names, output_folder, filename):
    # Define the bins and labels
    bins = [0, 50, 100, 200, 400, 700, 1000, 1500, float('inf')]
    labels = ['<50', '50-100', '100-200', '200-400', '400-700', '700-1000', '1000-1500', '1500>']
    # Initialize the dictionary
    profession_counts = {}
    for df, group_name in zip(groups_data, group_names):
        # Convert 'posts_count' to numeric, coercing errors to NaN
        df = df.copy()  # Ensure we're working on a copy
        df.loc[:, 'statuses_count'] = pd.to_numeric(df['statuses_count'], errors='coerce')  
        # Drop rows with NaN values in 'posts_count'
        df = df.dropna(subset=['statuses_count'])
        # Categorize posts count using consistent binning
        df['posts_count_range'] = pd.cut(df['statuses_count'], bins=bins, labels=labels)
        # Count the number in each range
        df_range_counts = df['posts_count_range'].value_counts().sort_index()
        profession_counts[group_name] = df_range_counts

    # Combine the counts into a DataFrame
    combined_counts = pd.DataFrame(profession_counts)

    # Plot the combined data
    combined_counts.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
    plt.title('Number of posts Count Range for each populations')
    plt.xlabel('posts Count Range')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=0)
    plt.legend(title='Group')
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path)
    plt.close()

def plot_top_authors_by_tweet_count(df, group_name, output_folder):
    """Plot top authors by tweet count."""
    author_counts = df['author'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=author_counts.index, y=author_counts.values, palette='viridis')
    plt.title(f"Top Authors by Tweet Count in {group_name}")
    plt.xlabel("Author")
    plt.ylabel("Tweet Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{group_name}_top_authors.png"))
    plt.close()

def plot_top_locations_by_count(df, group_name, output_folder):
    """
    Plot top 10 countries by number of occurrences.
    """
    # Count the occurrences of each country and get the top 10
    # df = df[df['country'] != 'NaN']
    country_counts = df['country'].value_counts().head(10)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=country_counts.index, y=country_counts.values, palette='viridis')
    plt.title(f"Top 10 Countries by Count in {group_name}")
    plt.xlabel("Country")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, f"{group_name}_top_countries.png"))
    plt.close()

def plot_scatter_with_limits_multiple_groups(data_groups, labels, x_column, y_column, x_limit, y_limit, title, xlabel, ylabel, output_folder, sizes=None, colors=None, markers=None):
    """
    Creates a scatter plot with custom settings for multiple datasets.
    """
    plt.figure(figsize=(10, 6))

    if sizes is None:
        sizes = [50] * len(data_groups)
    if colors is None:
        colors = ['blue', 'green', 'red', 'orange', 'purple'][:len(data_groups)]
    if markers is None:
        markers = ['o', '^', 's', 'p', '*'][:len(data_groups)]

    for i, data in enumerate(data_groups):
        plt.scatter(data[x_column], data[y_column], color=colors[i], edgecolor='black', label=labels[i], alpha=0.7, s=sizes[i], marker=markers[i])

    plt.xlim(0, x_limit)
    plt.ylim(0, y_limit)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if len(labels) > 1:
        plt.legend(title='Group', loc='upper left')
    plt.grid(True)
    file_path = os.path.join(output_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(file_path)
    plt.close()

def analyze_groups_with_scatter(data_groups, group_names):
    """
    Analyze multiple groups and include scatter plot for comparison.
    """
    for df, group_name in zip(data_groups, group_names):
        analyze_group(df, group_name)

    plot_scatter_with_limits_multiple_groups(
        data_groups=data_groups,
        labels=group_names,
        x_column='followers_count',
        y_column='statuses_count',
        x_limit=20000,
        y_limit=20000,
        title='Followers vs Posts: Comparison between Groups',
        xlabel='Followers Count',
        ylabel='Posts Count'
    )

def plot_tweets_distribution_per_month_merged(dataframes, group_labels, title, output_folder):
    """Plot merged distribution of tweets per month for multiple groups."""
    combined_df = pd.concat(dataframes, keys=group_labels).reset_index(level=0).rename(columns={'level_0': 'Group'})
    if 'date' not in combined_df.columns:
        raise ValueError("Column 'date' is missing in the combined DataFrame.")

    combined_df.loc[:, 'month'] = pd.to_datetime(combined_df['date'].str.split().str[0], errors='coerce').dt.to_period('M')

    tweets_per_month = combined_df.groupby(['month', 'Group']).size().unstack()
    plt.figure(figsize=(20, 8))
    tweets_per_month.plot(kind='bar', stacked=False, figsize=(20, 8), legend=(len(group_labels) > 1))
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    if len(group_labels) > 1:
        plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    save_path = os.path.join(output_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

def save_statistics_to_csv(data, group_name, summary_file):
    """Save statistics of a group's data to a single CSV file."""
    statistics = []
    for column in ['friends_count', 'followers_count', 'statuses_count']:
        if column in data.columns:
            description = describe_column(data, column).to_frame().T
            description['group'] = group_name
            description['column'] = column
            statistics.append(description)

    if statistics:
        combined_statistics = pd.concat(statistics, ignore_index=True)
        combined_statistics = combined_statistics.iloc[:,::-1]
        if not os.path.exists(summary_file):
            combined_statistics.to_csv(summary_file, index=False)
        else:
            combined_statistics.to_csv(summary_file, mode='a', index=False, header=False)


def main():
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    summary_file = os.path.join(output_folder, 'summary_statistics.csv')
    BINS = [0, 100, 500, 1000, 5000, 10000, 20000]
    LABELS = ['0-100', '101-500', '501-1000', '1001-5000', '5001-10000', '10001-20000']

    if len(sys.argv) < 2:
        print("Error: Please provide the path to the database file as a command-line argument.")
        return
    
    db_path = sys.argv[1]
    
    if not os.path.exists(db_path):
        print(f"Error: The database file '{db_path}' does not exist.")
        return
    conn = sqlite3.connect(db_path)

    authors_table = 'authors'
    posts_table = 'posts'

    try:
        author_columns = pd.read_sql(f"PRAGMA table_info({authors_table})", conn)['name'].tolist()
        if 'label' in author_columns:
            all_author_data = read_db(conn, authors_table)
            unique_labels = all_author_data['label'].unique()

            group_data = []
            group_names = []

            for label in unique_labels:
                # Skip invalid labels (None or NaN)
                if label is None or pd.isna(label):
                    print("Skipping invalid group label: None or NaN")
                    continue

                group = all_author_data.query(f"label == '{label}'")

                # Skip empty groups
                if group.empty:
                    print(f"Skipping empty group for label: {label}")
                    continue

                group_data.append(group)
                group_names.append(label)

            analyze_groups(group_data, group_names, output_folder, BINS, LABELS, summary_file)

            posts_data = read_db(conn, posts_table)


            # Process and visualize each group's tweet distribution
            for group, name in zip(group_data, group_names):
                group_posts = posts_data[posts_data['author'].isin(group['author_screen_name'].unique())].copy()
                group_posts['date_new'] = pd.to_datetime(
                    group_posts['date'].apply(lambda d: d.split()[0]),
                    errors='coerce'
                )
                group_posts['month'] = group_posts['date_new'].dt.to_period('M')

                distribution_path = os.path.join(output_folder, f"{name}_tweet_distribution.png")
                plot_tweet_distribution(
                    group_posts,
                    f"{name} Tweet Distribution",
                    f"{name}_tweet_distribution.png",
                    output_folder
                )
                plot_top_authors_by_tweet_count(group_posts, name, output_folder)


            plot_friends_count_range(group_data, group_names, output_folder, f"Friends_Count_Range_Merged.png")
            plot_followers_count_range(group_data, group_names, output_folder, f"Followers_Count_Range_Merged.png")
            plot_posts_count_range(group_data, group_names, output_folder, f"Posts_Count_Range_Merged.png")

            # Combined distribution plot
            plot_tweets_distribution_per_month_merged(
                [posts_data[posts_data['author'].isin(group['author_screen_name'].unique())] for group in group_data],
                group_names,
                "Tweet Distribution by Month for All Groups",
                output_folder
            )

            # Combined scatter plot
            plot_scatter_with_limits_multiple_groups(
                group_data,
                group_names,
                x_column='followers_count',
                y_column='statuses_count',
                x_limit=20000,
                y_limit=20000,
                title="Followers Count vs. Posts Count for All Groups",
                xlabel='Followers Count',
                ylabel='Post Count',
                output_folder=output_folder
            )
        else:
            all_data = read_db(conn, authors_table)

            analyze_group(all_data, "All_Populations", output_folder, BINS, LABELS, summary_file)

            # Plot tweet distribution
            posts_data = read_db(conn, posts_table)
            posts_data['date_new'] = pd.to_datetime(posts_data['date'].apply(lambda d: d.split()[0]), errors='coerce')
            posts_data['month'] = posts_data['date_new'].dt.to_period('M')
            plot_tweet_distribution(posts_data, "All Populations Tweet Distribution", "All_Populations_tweet_distribution.png", output_folder)
            plot_top_authors_by_tweet_count(posts_data, "All_Populations", output_folder)

            # Combined distribution plot
            plot_tweets_distribution_per_month_merged(
                [posts_data],
                ["All_Populations"],
                "Tweet Distribution by Month for All Populations",
                output_folder
            )
            
            # Combined scatter plot
            plot_scatter_with_limits_multiple_groups(
                [all_data],
                ["All_Populations"],
                x_column='followers_count',
                y_column='statuses_count',
                x_limit=20000,
                y_limit=20000,
                title="Followers Count vs. Posts Count for All Populations",
                xlabel='Followers Count',
                ylabel='Post Count',
                output_folder=output_folder
            )


        # script_path = "Llama_3_Hugging_Face_Cleaned.py"

        # try:
        #   # Run the Python script with the db_path as an argument
        #   result = subprocess.run(
        #       ["python", script_path, db_path],
        #       capture_output=True,
        #       text=True
        #   )
  
        #   # Check the return code
        #   if result.returncode != 0:
        #       raise RuntimeError(
        #           f"Subprocess failed with return code {result.returncode}.\n"
        #           f"Error Output:\n{result.stderr}"
        #       )
        #   else:
        #       print("Subprocess completed successfully.")
        #       print("Output:")
        #       print(result.stdout)

        # except FileNotFoundError:
        #     print(f"Error: The script {script_path} was not found.")
        # except RuntimeError as e:
        #     print(f"Subprocess Error: {e}")
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")

        if 'label' in author_columns:
            all_author_data = read_db(conn, authors_table)
            unique_labels = all_author_data['label'].unique()
            group_data = []
            group_names = []
            for label in unique_labels:
                # Skip invalid labels (None or NaN)
                if label is None or pd.isna(label):
                    print("Skipping invalid group label: None or NaN")
                    continue
                group = all_author_data.query(f"label == '{label}'")
                # Skip empty groups
                if group.empty:
                    print(f"Skipping empty group for label: {label}")
                    continue
                group_data.append(group)
                group_names.append(label)

            # Process and visualize each group's tweet distribution
            for group, name in zip(group_data, group_names):
                plot_top_locations_by_count(group, name, output_folder)
        else:
            plot_top_locations_by_count(all_data, "All_Populations", output_folder)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()



