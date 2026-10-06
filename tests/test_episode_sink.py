import threading
from types import SimpleNamespace
from expo_ft.utils.loop_utils import EpisodeSink


def test_outcome_labels_previous_action_and_terminal_flush_has_no_extra_transition():
    inserted=[]
    sink=EpisodeSink(SimpleNamespace(insert_transition=inserted.append),threading.Lock(),None,None,None,
        save_buffer=False,checkpoint_model=False,checkpoint_interval=100,start_step=0)
    sink.record_transition(0,{'observations':'s0','actions':'u0'})
    sink.flush_transitions()
    assert not inserted
    sink.label_last(0.,1.,False)  # result s1 of u0
    sink.record_transition(1,{'observations':'s1','actions':'u1'})
    sink.flush_transitions()
    assert inserted==[{'observations':'s0','actions':'u0','rewards':0.,'masks':1.,'dones':False}]
    sink.label_last(1.,0.,True)  # terminal s2 of u1; do not record s2/u2
    sink.flush_transitions()
    assert len(inserted)==2 and inserted[-1]['actions']=='u1' and inserted[-1]['dones']
    assert inserted[-1]['rewards']==1. and not sink._transitions
