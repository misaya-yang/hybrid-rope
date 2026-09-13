"""Exact real-token accounting for the locked 8K,8K,16K training cycle."""

SCHEDULE=(8192,8192,16384)
CYCLE_TOKENS=sum(SCHEDULE)

def _nonnegative(value,name):
 value=int(value)
 if value<0:raise ValueError(name+' must be nonnegative')
 return value

def length_at(micro_sequence):
 return SCHEDULE[_nonnegative(micro_sequence,'micro_sequence')%len(SCHEDULE)]

def tokens_before(micro_sequences):
 count=_nonnegative(micro_sequences,'micro_sequences');cycles,remainder=divmod(count,len(SCHEDULE))
 return cycles*CYCLE_TOKENS+sum(SCHEDULE[:remainder])

def micro_sequences_for_tokens(tokens):
 target=_nonnegative(tokens,'tokens');cycles,remainder=divmod(target,CYCLE_TOKENS)
 if remainder:
  partial=0
  for index,length in enumerate(SCHEDULE,1):
   partial+=length
   if partial==remainder:return cycles*len(SCHEDULE)+index
  raise ValueError('CPT endpoint does not align with the locked length schedule')
 return cycles*len(SCHEDULE)

def occurrence_before(micro_sequence,length):
 count=_nonnegative(micro_sequence,'micro_sequence');cycles,remainder=divmod(count,len(SCHEDULE))
 if length not in SCHEDULE:raise ValueError('length is outside the locked schedule')
 return cycles*SCHEDULE.count(length)+SCHEDULE[:remainder].count(length)
