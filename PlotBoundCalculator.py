class PlotBoundCalculator:
    @staticmethod
    def calculate_distance_ratio(top_plots, bottom_plots, max_distance=30):
        if len(top_plots) <= 0 or len(bottom_plots) <= 0:
            raise Exception

        top_plot = top_plots[0]
        bottom_plot = PlotBoundCalculator.__get_bottom_plot(top_plot, bottom_plots, max_distance)
        top_distance = PlotBoundCalculator.calculate_distance(top_plot)
        bottom_distance = PlotBoundCalculator.calculate_distance(bottom_plot)
        ratio = bottom_distance / top_distance

        return (top_distance, bottom_distance, ratio)

    @staticmethod
    def __get_bottom_plot(top_plot, bottom_plots, distance):
        end_of_top_plot = top_plot[-1]

        for plot in bottom_plots:
            bottom_plot_start = plot[0]
            distance_between_top_bottom = abs(bottom_plot_start - end_of_top_plot)
            if distance_between_top_bottom <= distance:
                return plot

    @staticmethod
    def calculate_distance(plot):
        start = plot[0]
        end = plot[-1]
        return end - start