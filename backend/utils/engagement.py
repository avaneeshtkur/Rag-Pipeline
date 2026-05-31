def compute_engagement_rate(likes: int, comments: int, views: int) -> float:
    """
    Engagement Rate = (likes + comments) / views * 100
    Returns float rounded to 4 decimal places.
    Handles division by zero.
    """
    try:
        likes = int(likes) if likes else 0
        comments = int(comments) if comments else 0
        views = int(views) if views else 0
    except (ValueError, TypeError):
        return 0.0

    if views == 0:
        return 0.0
    return round(((likes + comments) / views) * 100, 4)
