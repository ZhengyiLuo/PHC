from lxml import etree
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from copy import deepcopy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--in-model', type=str, default='human36m_vis_single_v1')
parser.add_argument('--out-model', type=str, default='human36m_vis_sample_single_v1')
parser.add_argument('--num', type=int, default=10)
parser.add_argument('--trans-range', type=int, default=(-1, -1))

args = parser.parse_args()

in_file = 'assets/mujoco_models/%s.xml' % args.in_model
out_file = 'assets/mujoco_models/%s.xml' % args.out_model
parser = XMLParser(remove_blank_text=True)
tree = parse(in_file, parser=parser)
root = tree.getroot().find('worldbody')
body = root.find('body')
for i in range(1, args.num):
    new_body = deepcopy(body)
    if args.trans_range[0] <= i < args.trans_range[1]:
        new_body.attrib['childclass'] = 'trans'
    new_body.attrib['name'] = '%d_%s' % (i, new_body.attrib['name'])
    for node in new_body.findall(".//body"):
        node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
    for node in new_body.findall(".//joint"):
        node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
    root.append(new_body)
if args.trans_range[0] == 0:
    body.attrib['childclass'] = 'trans'

tree.write(out_file, pretty_print=True)