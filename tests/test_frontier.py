import xml.etree.ElementTree as ET

from arlab.frontier import FrontierPoint, frontier_svg


def test_frontier_is_valid_svg_with_a_dot_and_label_per_point_in_each_panel():
    points = [FrontierPoint("a", 0.5, 0.2, 30.0), FrontierPoint("b", 0.3, 0.05, 10.0)]
    root = ET.fromstring(frontier_svg(points))
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}circle")) == 4
    labels = [t.text for t in root.findall(f"{ns}text") if t.get("class") == "label"]
    assert sorted(labels) == ["a", "a", "b", "b"]
