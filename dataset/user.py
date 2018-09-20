from itertools import islice, repeat, chain
from yxt_nlp_toolkit.utils import tokenizer

nil_token = '<NIL>'


class User:
    def __init__(self,
                 org='',
                 industry_l1_name='',
                 industry_l2_name='',
                 position='',
                 department='',
                 job_function=None):
        self.org = org
        self.industry_l1 = industry_l1_name
        self.industry_l2 = industry_l2_name
        self.position = position
        self.department = department
        self.job_function = job_function

    def __repr__(self):
        def _segment(field, value):
            if not value:
                return ''
            return "{k}='{v}'".format(k=field, v=value)

        fields = (_segment('org', self.org),
                  _segment('industry_l1', self.industry_l1),
                  _segment('industry_l2', self.industry_l2),
                  _segment('position', self.position),
                  _segment('department', self.department),
                  _segment('job_function', self.job_function))
        return 'User({kwargs})'.format(
            kwargs=','.join(e for e in fields if e)
        )

    @staticmethod
    def _regular_tokens_of(text, fixed_len):
        tokens = tokenizer(text, skip_space=True)
        if fixed_len <= 0:
            yield from tokens
        else:
            yield from islice(chain(tokens, repeat(nil_token)), 0, fixed_len)

    def token_seq(self, fixed_len=0):
        fields = (self.org, self.industry_l1, self.industry_l2, self.position, self.department)
        # fields = (self.position, self.department)
        tokens = (self._regular_tokens_of(e, fixed_len=fixed_len) for e in fields)
        return tuple(chain(*tokens))
