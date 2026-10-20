from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    TEXT_BLOCK = "TextBlock"
    FIGURE = "Figure"
    TABLE = "Table"
    EQUATION = "Equation"
    CONCEPT = "Concept"
    CLAIM = "Claim"
    PAPER = "Paper"


class EdgeType(str, Enum):
    CONTAINS = "CONTAINS"
    HAS_CAPTION = "HAS_CAPTION"
    REFERENCES = "REFERENCES"
    ILLUSTRATES = "ILLUSTRATES"
    QUANTIFIES = "QUANTIFIES"
    DEFINES = "DEFINES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DERIVES_FROM = "DERIVES_FROM"
    COMPARES = "COMPARES"
    SAME_CONCEPT = "SAME_CONCEPT"


@dataclass
class KGNode:
    node_id: str
    node_type: NodeType
    paper_id: str
    page: int = -1
    content: str = ""
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "paper_id": self.paper_id,
            "page": self.page,
            "content": self.content,
            "image_path": self.image_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> KGNode:
        d["node_type"] = NodeType(d["node_type"])
        return cls(**d)


@dataclass
class KGEdge:
    src_id: str
    dst_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> KGEdge:
        d["edge_type"] = EdgeType(d["edge_type"])
        return cls(**d)


@dataclass
class ParsedElement:
    paper_id: str
    page: int
    element_type: str
    content: str
    image_path: Optional[str]
    confidence: float
    caption: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
