from src.causalaibook.editor.classes.options_parser import OptionsParser
from src.causalaibook.selection_bias.classes.selection_bias import selectionBiasNodeType


class SelectionBiasOptionsParser(OptionsParser):

    def getType(self):
        return selectionBiasNodeType


    def toString(self, target, graph):
        return ''


    def fromString(self, text, target, graph):
        pass