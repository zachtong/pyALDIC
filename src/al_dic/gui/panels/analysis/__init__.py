"""The strain window's Analysis tab, in parts.

``tab``            the coordinator: engine, caches, probe edits, frames, exports
``canvas_panel``   tools, the probe canvas, the field under it, the navigator
``chart_panel``    views, plot controls and the chart
``probe_table``    the probe list
``quantities``     what can be plotted and how it is scaled (no widgets)
``text``           the tab's translated words, all in context "AnalysisTab"
"""

from al_dic.gui.panels.analysis.tab import AnalysisTab

__all__ = ["AnalysisTab"]
