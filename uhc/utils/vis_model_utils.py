from lxml import etree
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from copy import deepcopy


def create_vis_model(in_file, out_file, num=10):
    xml_parser = XMLParser(remove_blank_text=True)
    tree = parse(in_file, parser=xml_parser)
    # remove_elements = ['actuator', 'contact', 'equality', 'sensor']
    remove_elements = ['actuator', 'contact', 'equality']
    for elem in remove_elements:
        node = tree.getroot().find(elem)
        if node is None:
            print(f"has no elem: {elem}")
        else:
            node.getparent().remove(node)
    
    option = tree.getroot().find('option')
    flag = SubElement(option, 'flag', {'contact': 'disable'})
    option.addnext(Element('size', {'njmax': '1000'}))

    worldbody = tree.getroot().find('worldbody')
    body = worldbody.find('body')
    for i in range(1, num):
        new_body = deepcopy(body)
        new_body.attrib['name'] = '%d_%s' % (i, new_body.attrib['name'])
        for node in new_body.findall(".//body"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//joint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//site"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        worldbody.append(new_body)
    tree.write(out_file, pretty_print=True)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--in_model', type=str, default='assets/mujoco_models/models/character1/model.xml')
    parser.add_argument('--out_model', type=str, default='assets/mujoco_models/models/character1/model_vis.xml')
    args = parser.parse_args()

    in_model = f'assets/mujoco_models/models/{args.cfg}/model.xml' if args.cfg is not None else args.in_model
    out_model = f'assets/mujoco_models/models/{args.cfg}/model_vis.xml' if args.cfg is not None else args.out_model

    create_vis_model(in_model, out_model)