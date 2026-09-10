import json
import unittest

from experiments.refcarry_audit.prepare_mrcr_pairs import family, locate, ordinal


class MRCRPairChecks(unittest.TestCase):
    def row(self):
        messages = [{'role': 'user', 'content': 'Recall prior answers exactly.'}]
        for value in ('alpha answer', 'beta answer', 'gamma answer'):
            messages.extend([{'role': 'user', 'content': 'Write a poem about maps.'},
                             {'role': 'assistant', 'content': value}])
        messages.append({'role': 'user', 'content': 'Prepend ABC123 to the 2nd (1 indexed) poem about maps. No other text.'})
        return {'prompt': json.dumps(messages), 'answer': 'ABC123beta answer',
                'random_string_to_prepend': 'ABC123', 'n_needles': 3,
                'desired_msg_index': 3}

    def test_counterfactual_labels_exchange_without_query_changes(self):
        records = family(self.row(), 42)
        self.assertEqual(len(records), 6)
        a, b, c, d = records[:4]
        self.assertEqual(a['messages'][-1], c['messages'][-1])
        self.assertEqual(b['messages'][-1], d['messages'][-1])
        self.assertEqual(a['references'], d['references'])
        self.assertEqual(b['references'], c['references'])
        self.assertNotEqual(a['references'], b['references'])
        self.assertEqual(a['history_sha256'], b['history_sha256'])
        self.assertEqual(c['history_sha256'], d['history_sha256'])
        self.assertNotEqual(a['history_sha256'], c['history_sha256'])

    def test_compact_keeps_all_competing_answers(self):
        records = family(self.row(), 42)
        for record in records[-2:]:
            bodies = [m['content'] for m in record['messages'] if m['role'] == 'assistant']
            self.assertEqual(bodies, ['alpha answer', 'beta answer', 'gamma answer'])
            target = record['messages'][record['target_message_index']]['content']
            self.assertEqual(record['references'], ['ABC123'+target])

    def test_corrupt_original_label_and_metadata_rejected(self):
        for change in ({'answer': 'ABC123wrong'}, {'desired_msg_index': 4}, {'n_needles': 2}):
            row = self.row();row.update(change)
            with self.assertRaises(ValueError):locate(row)

    def test_ordinals(self):
        self.assertEqual([ordinal(n) for n in (1, 2, 3, 11, 12, 13, 21)],
                         ['1st', '2nd', '3rd', '11th', '12th', '13th', '21st'])


if __name__ == '__main__':
    unittest.main()
