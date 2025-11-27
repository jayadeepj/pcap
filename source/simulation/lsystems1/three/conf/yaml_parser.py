import yaml

yaml_file = None


class LSystemConfig:
    """ Axioms, rules and alphabets for a branch L-System"""

    def __init__(self):
        # L-System Configuration
        self.axiom = None
        self.n = None

        self.T = None
        self.e = None
        self.theta = None

        self.rules = []

        self.params = []

        self.free_params = {}

    class Rule:
        def __init__(self, pred, succ, cond=None):
            self.pred = pred
            self.succ = succ
            self.cond = cond

    def add_rule(self, pred, succ, cond):
        rule = self.Rule(pred, succ, cond)
        self.rules.append(rule)

    class Param:
        def __init__(self, operand, ident):
            """ F(c) => F is the operand and c is the ident, initial value is taken from the axiom """
            self.operand = operand
            self.ident = ident

    def add_param(self, operand, ident):
        param = self.Param(operand=operand, ident=ident)
        self.params.append(param)


class TreeConfig:
    """ Axioms, rules and alphabets for a branch L-System"""

    def __init__(self):
        # L-System Configuration
        self.len_scale_factor = None
        self.rad_scale_factor = None

        self.density = None


def config():
    with open(yaml_path(), 'r') as stream:
        return yaml.safe_load(stream)


def yaml_path():
    assert yaml_file is not None, " Set the yaml path first"
    return yaml_file


def yaml_to_lsystem():
    yaml_config = config()['tree']['geometry']
    l_config = LSystemConfig()
    l_config.axiom = yaml_config['lsystem']['axiom']
    l_config.n = yaml_config['lsystem']['n']

    l_config.T = yaml_config['lsystem']['T']
    l_config.e = yaml_config['lsystem']['e']
    l_config.theta = yaml_config['lsystem']['theta']
    l_config.sigma = yaml_config['lsystem']['sigma']

    free_params = yaml_config['lsystem'].get('free_params', {})
    l_config.free_params = {param_name: param_value
                            for p in free_params for param_name, param_value in p.items()}

    l_config.randomise = yaml_config['lsystem'].get('randomise', False)
    l_config.randomise_cnt = yaml_config['lsystem'].get('randomise_cnt', 1)
    l_config.rel_std = yaml_config['lsystem'].get('rel_std', 0.1)

    for rule_data in yaml_config['lsystem']['rules']:
        l_config.add_rule(rule_data['pred'], rule_data['succ'], rule_data['cond'])

    for param_data in yaml_config['lsystem']['params']:
        l_config.add_param(param_data['operand'], param_data['ident'])

    return l_config


def yaml_to_tree_config():
    yaml_config = config()['tree']['geometry']
    t_config = TreeConfig()
    t_config.density = yaml_config['real']['physics']['density']
    t_config.len_scale_factor = yaml_config['real']['len_scale_factor']
    t_config.rad_scale_factor = yaml_config['real']['rad_scale_factor']
    dof_root = yaml_config['post']['dof_root']
    t_config.dof_root = dof_root.replace('link-', '') if dof_root is not None else ''
    return t_config


def out_file():
    yaml_config = config()['tree']
    return yaml_config['outfile']
