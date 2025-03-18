"""
@author: akalenkova (anna.kalenkova@adelaide.edu.au)
"""

import math

class Gauss:
    def __init__(self, mean, deviation):
        self.mean = mean 
        self.deviation = deviation 


    """
    Calculate the areaunder the curve when x < 0
    """
    def calc_negative_area(self):
        cdf = 0.5 * (1 + math.erf(-self.mean/(self.deviation*math.sqrt(2))))
        return cdf

    def to_json(self):
        return {'mean': self.mean, 'deviation': self.deviation}
