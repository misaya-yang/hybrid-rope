import unittest

from experiments.refcarry_audit.score_mrcr_pairs import aggregate, score_text


class ScoreChecks(unittest.TestCase):
    def setUp(self):
        self.row = dict(random_string_to_prepend='Q7Z',references=['Q7Zcorrect body'],
            messages=[{'content':'correct body'}, {'content':'other body'}],
            occurrence_message_indices=[0,1])

    def test_no_substring_whitespace_or_missing_eos_promotion(self):
        self.assertTrue(score_text('Q7Zcorrect body', True, self.row)['full_exact_and_eos'])
        for text, eos in [('Q7Zcorrect body',False), (' Q7Zcorrect body',True),
                          ('Q7Zcorrect body\nextra explanation',True)]:
            self.assertFalse(score_text(text,eos,self.row)['full_exact_and_eos'])
        self.assertEqual(score_text('correct body',True,self.row)['official_sequence_ratio'],0.)

    def test_wrong_occurrence_is_separate_from_partial_similarity(self):
        result = score_text('Q7Zother body',True,self.row)
        self.assertTrue(result['wrong_occurrence_whole_string'])
        self.assertFalse(result['whole_string_exact'])
        self.assertGreater(result['official_sequence_ratio'],0.)

    def test_family_success_requires_all_counterfactuals(self):
        rows = []
        for i in range(4):
            rows.append(dict(method='x',family_id='one',variant='counterfactual',
                **score_text('Q7Zcorrect body' if i<3 else 'Q7Zother body',True,self.row)))
        result = aggregate(rows)['x']
        self.assertEqual(result['complete_four_way_families'],1)
        self.assertEqual(result['four_way_exact_and_eos'],0)


if __name__ == '__main__':unittest.main()
