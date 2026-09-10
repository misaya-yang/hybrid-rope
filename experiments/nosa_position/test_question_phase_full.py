from .question_phase_full import MARKER,QuestionPhaseSelector,question_boundary
from .selector_controls import BlockSummarySelector

def test_boundary_and_registration():
    text='Context.\n'+MARKER+'some-key?'
    row={'prompt':text,'prompt_ids':list(range(len(text)))}
    def tokenizer(value,**kwargs):
        assert value==text
        return {'input_ids':list(range(len(text))),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    receipt=question_boundary(row,tokenizer)
    assert receipt['switch_query_position']==len('Context.\n')
    assert receipt['full_scored_prompt_tokens']==len(MARKER+'some-key?')
    assert isinstance(QuestionPhaseSelector,type)
    selector=QuestionPhaseSelector('native_question_full')
    assert isinstance(selector,QuestionPhaseSelector) and isinstance(selector,BlockSummarySelector)

if __name__=='__main__':test_boundary_and_registration();print('PASS: exact question onset and class-based runner registration')
