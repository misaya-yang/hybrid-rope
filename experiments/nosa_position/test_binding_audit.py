from .binding_audit import inspect_answer

def test_explicit_binding():
    row={'prompt':'One of the special magic numbers for a-key is: 12. One of the special magic numbers for b-key is: 34.\nWhat are all the special magic numbers for a-key, and b-key mentioned in the provided text?', 'references':['34','12'],'row_id':'test'}
    def result(text):return {'selector':'test','output_text':text,'official_recall':1.,'ended_with_eos':True}
    good=inspect_answer(row,result('- a-key: 12\n- b-key: 34'))
    wrong=inspect_answer(row,result(': 34 and 12, respectively.'))
    unordered=inspect_answer(row,result('34, 12'))
    assert good['all_bindings_correct_plus_eos'] is True
    assert wrong['correctly_asserted_key_count']==0 and len(wrong['wrong_explicit_claims'])==2
    assert unordered['all_bindings_correct'] is None
    assert inspect_answer(row,{**result('12,34'),'ended_with_eos':False})['all_bindings_correct_plus_eos'] is False
    thousands=inspect_answer(row,result('12 and 1,234, respectively.'))
    extra=inspect_answer(row,result('12, 34 and 34, respectively.'))
    assert thousands['claims'][1]['value']=='1234' and len(thousands['wrong_explicit_claims'])==1
    assert inspect_answer(row,result('12,34 respectively.'))['all_bindings_correct_plus_eos'] is True
    assert extra['binding_cardinality_error']['asserted_values']==3 and extra['all_bindings_correct_plus_eos'] is False

if __name__=='__main__':test_explicit_binding();print('PASS: source-owned gold, explicit swaps detected, unordered answers left unknown')
