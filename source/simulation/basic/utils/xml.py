import json
from lxml import etree, objectify


def extract_support_mass_ratio(urdf_path):
    """ Deserialize support mass ratio from comments in urdf"""
    with open(urdf_path, "r") as f:
        urdf_content = f.read()

    commented_tree = objectify.fromstring(urdf_content)
    for node in commented_tree.iterchildren(etree.Comment):
        node_content = etree.tostring(node).decode("utf-8")
        if 'support_mass_ratio=' in node_content:
            cleaned_node_content = node_content \
                .replace("<!--", "") \
                .replace("-->", "") \
                .replace("support_mass_ratio=", "") \
                .strip()

            return json.loads(cleaned_node_content)


def extract_min_leaf_force(urdf_path):
    with open(urdf_path, "r") as f:
        urdf_content = f.read()

    commented_tree = objectify.fromstring(urdf_content)
    for node in commented_tree.iterchildren(etree.Comment):
        node_content = etree.tostring(node).decode("utf-8")
        if 'min_leaf_force=' in node_content:
            cleaned_node_content = node_content \
                .replace("<!--", "") \
                .replace("-->", "") \
                .replace("min_leaf_force=", "") \
                .strip()

    return float(cleaned_node_content)
