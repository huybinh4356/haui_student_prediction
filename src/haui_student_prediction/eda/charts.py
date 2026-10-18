from pathlib import Path
from haui_student_prediction.eda.distributions import DistributionPlots
from haui_student_prediction.eda.relationships import RelationshipPlots
from haui_student_prediction.eda.statistics import EDAStatistics


class ChartService:
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dist_plots = DistributionPlots()
        self.relationship_plots = RelationshipPlots()
        self.statistics = EDAStatistics()
