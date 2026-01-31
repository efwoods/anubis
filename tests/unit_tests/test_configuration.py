from langgraph.pregel import Pregel

from src.subgraphs.conversational_memory_graph.graph import graph


def test_placeholder() -> None:
    # TODO: You can add actual unit tests
    # for your graph and other logic here.
    assert isinstance(graph, Pregel)
