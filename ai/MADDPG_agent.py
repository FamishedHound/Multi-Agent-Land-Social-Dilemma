class MADDPGAGENT():
    def select_action(self, neural_net_output_number):
        a_bag_numbers = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        return min(range(len(a_bag_numbers)), key=lambda i: abs(a_bag_numbers[i] - neural_net_output_number))
