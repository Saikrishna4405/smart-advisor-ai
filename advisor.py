def investment_decision(volatility, avg_return, trend):

    score = 0

    if avg_return > 0:
        score += 1
    else:
        score -= 1

    if trend > 0:
        score += 1
    else:
        score -= 1

    if volatility < 0.02:
        score += 1
    else:
        score -= 1

    if score >= 2:
        recommendation = "Strong Buy"
    elif score == 1:
        recommendation = "Buy"
    elif score == 0:
        recommendation = "Hold"
    else:
        recommendation = "Avoid"

    return recommendation, score
