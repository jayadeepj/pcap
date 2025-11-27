from simulation.lsystems1.three.fractal import rule_parser as parser


def eval_axiom(l_config):
    axiom = l_config.axiom

    for param in l_config.params:
        if param.ident in axiom:
            axiom = axiom.replace(param.ident, str(param.initial))
    return axiom


def _evaluate(_succ, _whole_params):
    temp_eval_str = ""
    _succ_tokens = parser.tokenize(_succ)

    for _token in _succ_tokens:
        _operands = parser.extract_operand(_token)
        if "(" not in _operands or ")" not in _operands:
            temp_eval_str += _operands
        else:
            _evals = parser.extract_evals(_token)
            _eval_res = [eval(e, _whole_params) for e in _evals]
            _eval_succ = _operands.format(','.join([str(er) for er in _eval_res]))
            temp_eval_str += _eval_succ
    return temp_eval_str


def gen_lsystem(l_config):
    """ L-System recurser that creates the expanded string"""
    l_string = l_config.axiom
    l_strings = [l_string]
    max_print_len = 300  # truncate the l_string for print.

    for _ in range(l_config.n):
        l_tokens = parser.tokenize(l_string)
        new_l_string = ''
        for token in l_tokens:
            rule_matched = False
            for rule in l_config.rules:

                partial_match, match_params = parser.check_format(rule.pred, token)
                whole_params = {**match_params, **l_config.free_params}

                if token == rule.pred:
                    assert rule.cond is None, f"No conditions allowed for plain match {token}, {rule.cond}"
                    eval_succ = _evaluate(rule.succ, whole_params)
                    new_l_string += eval_succ
                    rule_matched = True
                    break

                if partial_match:
                    if rule.cond is None or eval(rule.cond, whole_params) is True:
                        eval_succ = _evaluate(rule.succ, whole_params)
                        new_l_string += eval_succ
                        rule_matched = True

            if not rule_matched:
                new_l_string += token

        l_string = new_l_string
        l_strings.append(l_string)

        # printable = l_string[:max_print_len] if len(l_string) > max_print_len else l_string
        # print(f"Generation: {_ + 1}, String: {printable}")

    return l_strings
