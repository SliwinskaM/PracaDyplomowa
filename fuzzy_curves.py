# Parent class for fuzzy curves
class FuzzyCurves():
    def __init__(self, min_score, max_score, list_of_curves):
        self.min_score = min_score
        self.max_score = max_score
        self._list_of_curves = list_of_curves

    def get_number_of_sets(self):
        return len(self._list_of_curves)

    def get_list_of_curves(self):
        return self._list_of_curves


# Three curves
class Curves1(FuzzyCurves):
    def __init__(self, min_score, max_score):
        super().__init__(min_score, max_score, [self.low_curve, self.medium_curve, self.high_curve])

    def low_curve(self, score):
        # normalize the score to scale 0-1
        score_norm = (score - self.min_score) / (self.max_score - self.min_score)
        if score_norm <= 0.2:
            return 1
        if 0.2 < score_norm < 0.45:
            return round(1 - 4 * (score_norm - 0.2), 2)
        return 0

    def medium_curve(self, score):
        # normalize the score to scale 0-1
        score_norm = (score - self.min_score) / (self.max_score - self.min_score)
        if 0.2 < score_norm < 0.45:
            return round(4 * (score_norm - 0.2), 2)
        if 0.45 <= score_norm <= 0.55:
            return 1
        if 0.55 < score_norm < 0.8:
            return round(1 - 4 * (score_norm - 0.55), 2)
        return 0

    def high_curve(self, score):
        # normalize the score to scale 0-1
        score_norm = (score - self.min_score) / (self.max_score - self.min_score)
        if 0.55 < score_norm < 0.8:
            return round(4 * (score_norm - 0.55), 2)
        if score_norm >= 0.8:
            return 1
        return 0


# Two curves
class Curves2(FuzzyCurves):
    def __init__(self, min_score, max_score):
        super().__init__(min_score, max_score, [self.low_curve, self.high_curve])

    def low_curve(self, score):
        # normalize the score to scale 0-1
        score_norm = (score - self.min_score) / (self.max_score - self.min_score)
        return 1 - score_norm

    def high_curve(self, score):
        # normalize the score to scale 0-1
        score_norm = (score - self.min_score) / (self.max_score - self.min_score)
        return score_norm

