"""
Analyze the olist_order_reviews_dataset to check how many reviews have actual text comments.
"""

import pandas as pd
import sys

# Load the reviews dataset
print("Loading olist_order_reviews_dataset.csv...")
df = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")

print(f"\n📊 Reviews Dataset Analysis")
print("=" * 60)

# Total reviews
total_reviews = len(df)
print(f"\n✅ Total Reviews: {total_reviews:,}")

# Check review_comment_title column
if "review_comment_title" in df.columns:
    titles_not_null = df["review_comment_title"].notna().sum()
    titles_with_text = (
        df["review_comment_title"].fillna("").str.strip().str.len().gt(0).sum()
    )
    print(f"\n📝 Review Titles:")
    print(
        f"   - Not NULL: {titles_not_null:,} ({titles_not_null/total_reviews*100:.1f}%)"
    )
    print(
        f"   - With actual text: {titles_with_text:,} ({titles_with_text/total_reviews*100:.1f}%)"
    )

# Check review_comment_message column
if "review_comment_message" in df.columns:
    messages_not_null = df["review_comment_message"].notna().sum()
    messages_with_text = (
        df["review_comment_message"].fillna("").str.strip().str.len().gt(0).sum()
    )

    # Different thresholds for meaningful text
    messages_short = (
        df["review_comment_message"].fillna("").str.strip().str.len().gt(5).sum()
    )  # >5 chars
    messages_medium = (
        df["review_comment_message"].fillna("").str.strip().str.len().gt(10).sum()
    )  # >10 chars
    messages_long = (
        df["review_comment_message"].fillna("").str.strip().str.len().gt(50).sum()
    )  # >50 chars

    print(f"\n💬 Review Comments (Messages):")
    print(
        f"   - Not NULL: {messages_not_null:,} ({messages_not_null/total_reviews*100:.1f}%)"
    )
    print(
        f"   - With any text: {messages_with_text:,} ({messages_with_text/total_reviews*100:.1f}%)"
    )
    print(
        f"   - > 5 characters: {messages_short:,} ({messages_short/total_reviews*100:.1f}%)"
    )
    print(
        f"   - > 10 characters: {messages_medium:,} ({messages_medium/total_reviews*100:.1f}%)"
    )
    print(
        f"   - > 50 characters: {messages_long:,} ({messages_long/total_reviews*100:.1f}%)"
    )

# Reviews with either title or message
if "review_comment_title" in df.columns and "review_comment_message" in df.columns:
    has_title = df["review_comment_title"].fillna("").str.strip().str.len().gt(0)
    has_message = df["review_comment_message"].fillna("").str.strip().str.len().gt(0)
    has_any_text = (has_title | has_message).sum()

    print(f"\n🎯 Reviews with ANY text (title OR message):")
    print(f"   - Count: {has_any_text:,} ({has_any_text/total_reviews*100:.1f}%)")
    print(f"\n❌ Reviews with NO text at all:")
    print(
        f"   - Count: {total_reviews - has_any_text:,} ({(total_reviews - has_any_text)/total_reviews*100:.1f}%)"
    )

# Review score distribution
if "review_score" in df.columns:
    print(f"\n⭐ Review Score Distribution:")
    score_dist = df["review_score"].value_counts().sort_index()
    for score, count in score_dist.items():
        bar = "█" * int(count / total_reviews * 50)
        print(
            f"   {int(score)} star: {count:6,} ({count/total_reviews*100:5.1f}%) {bar}"
        )

# Sample reviews with text
if "review_comment_message" in df.columns:
    print(f"\n📖 Sample Reviews with Text:")
    print("=" * 60)

    sample_reviews = df[
        df["review_comment_message"].fillna("").str.strip().str.len().gt(20)
    ].head(3)

    for idx, row in sample_reviews.iterrows():
        score = int(row["review_score"]) if "review_score" in row else "N/A"
        message = (
            row["review_comment_message"][:150] + "..."
            if len(str(row["review_comment_message"])) > 150
            else row["review_comment_message"]
        )
        print(f"\n   Review #{idx + 1} | Score: {score}/5")
        print(f"   Message: {message}")

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
