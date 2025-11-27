import re


def tokenize(string):
    """ 'F(c)[+X]F(de)[-X]+X' => ['F(c)', '[', '+', 'X', ']', 'F(de)', '[', '-', 'X', ']', '+', 'X'] """
    pattern = r'[A-Z,+,/,-,&,^]\([^()]+\)|.'
    matches = re.findall(pattern, string)

    if not matches:
        return [string]

    return matches


def extract_operand(succ):
    """ 'F(c+1, w*2)' => 'F({})' """
    pattern = r'\([^()]+\)'
    replaced = re.sub(pattern, '({})', succ)
    return replaced


def extract_values(token):
    """ F(12,200) => [12, 200]"""
    pattern = r'\(([-?\d\s.,]+)\)'
    match = re.search(pattern, token)
    if match:
        values = match.group(1).split(',')
        return [float(value.strip()) for value in values]
    return None


def extract_idents(pred):
    """ F(c,w) => [c, w] """
    pattern = r'\(([\w\s,]+)\)'
    match = re.search(pattern, pred)
    if match:
        idents = match.group(1).split(',')
        return [ident.strip() for ident in idents]
    return None


def check_format(pred, token):
    """ Check if the pred & token formats match and if yes return the dict of values
        E.g. pred = 'F(c)', token = 'F(200.3)' =>  True, {'c': 200.3}
        E.g. pred = 'X(c,w)', token = 'F(12.23, 200.43)' =>  False, {} """

    idents = extract_idents(pred)
    values = extract_values(token)
    params = {}
    if idents is not None and values is not None:
        token_without_values = re.sub(r'\([\d\s.,]+\)', '', token)
        pred_without_idents = re.sub(r'\([\w\s,]+\)', '', pred)
        if pred_without_idents == token_without_values:
            if len(idents) != len(values):
                raise ValueError(f"Mismatch in the number of identifiers and values idents{idents}, values{values}")

            params = {ident: value for ident, value in zip(idents, values)}
            return True, params

    return False, params


def extract_evals(succ):
    """ Extract the evaluation string from the rule list
        E.g. F(c+1, w*2)" => ['c+1', 'w*2'] """
    pattern = r'\(([^()]+)\)'
    matches = re.findall(pattern, succ)
    evals = [expr.strip() for expr in matches[0].split(',')] if matches else []
    return evals