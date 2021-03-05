# Funkcje przynależności
################ FUZZY CURVES ##################################
def low_curve(score, min_score, max_score):
    # normalize the score to scale 0-1
    score_norm = (score - min_score) / (max_score - min_score)
    if score_norm <= 0.2:
        return 1
    if 0.2 < score_norm < 0.45:
        return round(1 - 4 * (score_norm - 0.2), 2)
    return 0


def medium_curve(score, min_score, max_score):
    # normalize the score to scale 0-1
    score_norm = (score - min_score) / (max_score - min_score)
    if 0.2 < score_norm < 0.45:
        return round(4 * (score_norm - 0.2), 2)
    if 0.45 <= score_norm <= 0.55:
        return 1
    if 0.55 < score_norm < 0.8:
        return round(1 - 4 * (score_norm - 0.55), 2)
    return 0


def high_curve(score, min_score, max_score):
    # normalize the score to scale 0-1
    score_norm = (score - min_score) / (max_score - min_score)
    if 0.55 < score_norm < 0.8:
        return round(4 * (score_norm - 0.55), 2)
    if score_norm >= 0.8:
        return 1
    return 0


