from emotion_hate_sentiment_analysis import main as main_emotion
from generic_code import main as main_generic

def main():
    # Ask the user to input the database path
    db_path = input("Enter the path to the database: ")

    print("Select the code you want to run:")
    print("1 - EDA")
    print("2 - Emotion, Hate, Sentiment Analysis")
    
    choice = input("Enter your choice number: ")
    
    if choice == "1":
        main_generic(db_path)  # Pass the user-provided db_path
    elif choice == "2":
        main_emotion(db_path)  # Pass the user-provided db_path
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()