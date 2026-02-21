"""
Market utility functions for event analysis and detection
"""


def detect_event_type(m_poly, clob_buckets):
    """
    Detect if event is short (<72h) or long (≥72h)

    Uses hybrid approach:
    1. Primary: Analyze bucket sizes (short=~24 tweets, long=~19 tweets)
    2. Fallback: Heuristic from count/hours ratio
    3. Default: 'long' (conservative)

    Returns: ('short' or 'long', bucket_size or None)
    """
    # Method 1: Bucket size analysis (most reliable)
    bucket_sizes = []
    for b in clob_buckets:
        if 'max' in b and 'min' in b and b['max'] < 99999:
            size = b['max'] - b['min'] + 1
            bucket_sizes.append(size)

    if bucket_sizes:
        avg_size = sum(bucket_sizes) / len(bucket_sizes)
        # Threshold: 21.5 (midpoint between 19 and 24)
        event_type = 'short' if avg_size > 21.5 else 'long'
        return event_type, avg_size

    # Method 2: Heuristic from count/hours (fallback)
    if m_poly:  # Check if m_poly is not None
        count = m_poly.get('count', 0)
        hours_left = m_poly.get('hours', 0)

        if count > 50 and hours_left > 0:
            # If we have significant tweets, we can estimate
            ratio = count / hours_left
            # Short events typically have denser activity
            if ratio > 2.5 and hours_left < 48:
                return 'short', None

    # Fallback: assume long (conservative, won't apply aggressive fixes)
    return 'long', None
