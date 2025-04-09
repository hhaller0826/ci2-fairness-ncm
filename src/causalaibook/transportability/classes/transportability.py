from src.causalaibook.graph.classes.graph_defs import NodeType, nodeTypeMap
from src.causalaibook.transportability.classes.population import Population


transportabilityNodeType = NodeType('tr', 'Transportability', 'TR')
nodeTypeMap[transportabilityNodeType.id_] = transportabilityNodeType

targetPopulation = Population('Target', '*')
targetPopulation.id_ = '*'